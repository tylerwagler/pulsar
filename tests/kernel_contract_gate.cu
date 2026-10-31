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
 *    5. idx-decode-bound pulsar_gpu_indexer_scores_decode_run_tensor over a bank holding ONLY its visible rows, flush
 *                       against the end of an allocation, with a 1024-wide scores row: no read past the visible rows
 *                       (it used to stage the whole 128-row tile), scores equal to a garbage-padded bank's.
 *    6. ring-store-descr the 0731 ring batch store: no descriptor and a full one run; positions without seq_id
 *                       refuses (seq_id was dereferenced unchecked).
 *    7. hc-fused-width  the fused HC split + norm runs n_embd 4096 and 5120 and refuses 4352 (a width between them used
 *                       to launch the 5120 unroll with dead groups).
 *    8. ple-order       the Qwen PLE launcher refuses a verify capture wider than its batch BEFORE queuing a kernel:
 *                       streams and conv state come back byte-identical (the gate / conv kernels used to run first).
 *    9. hc-expand-sizes the split expands refuse a block / add / residual / split operand one row short of out_hc's
 *                       rows (no operand was checked); full operands run.
 *   10. subview-dry     pulsar_tensor_subview carries the base's dry flag, format and element size (dry was unset).
 *   11. bf16-nan        pulsar_f32_to_bf16_rne over all 2^32 f32 patterns: every non-NaN keeps the old expression's
 *                       bits, every NaN stays NaN (the old rounding add turned some into Inf and wrapped others to 0).
 *   12. swiglu-nan      pulsar_swiglu_elem (now also the CUTLASS FFN MMA kernel's) on 2^26 hashed (g, u) pairs x 3
 *                       clamps: every non-NaN pair keeps the fminf form's bits, a NaN operand stays NaN.
 *
 *  Prints one line per case and KERNEL CONTRACT GATE PASS / FAIL. */
#include "pulsar_gpu.h"
#include "cuda/pulsar_cuda_arch.cuh"
#include "cuda/pulsar_cuda_internal.h"   /* PULSAR_IDX_MXFP4_HEADS_V4 */
#include "cuda/pulsar_cuda_mx.cuh"
#include "cuda/pulsar_cuda_qwen.h"

#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

/* ---- 5. idx-decode-bound --------------------------------------------------------------------------------------- */

/// Case 5: a decode run whose bank holds exactly its visible rows, placed flush against the END of a fresh 2 MiB
/// allocation, over a 1024-wide scores row (the tile reaches far past the visible rows): the launch must not fault,
/// and its scores must equal the same run over a bank padded with garbage past the visible rows.
static void case_idx_decode_bound(void) {
    if (!pulsar_cuda_block_scale_mma(pulsar_cuda_device_major())) {
        printf("  5 idx-decode-bound: no block-scaled MMA on this device -- the scorer refuses; not reachable here\n");
        return;
    }
    const uint32_t n_head = PULSAR_IDX_MXFP4_HEADS_V4, rb = PULSAR_MXKV_FP4_ROWBYTES(128u);
    const uint32_t ratio = 4u, run_pos0 = 37u * 4u, run_n = 1u, n_comp = 1024u;
    const uint32_t vis = (run_pos0 + run_n) / ratio;   /* 37 */
    uint64_t s = 0x9e3779b97f4a7c15ull;
    auto fill = [&s](std::vector<uint8_t> &b, uint32_t rows) {
        for (uint32_t r = 0; r < rows; r++)
            for (uint32_t i = 0; i < PULSAR_MXKV_FP4_ROWBYTES(128u); i++) {
                s = s * 6364136223846793005ull + 1442695040888963407ull;
                b[(size_t)r * PULSAR_MXKV_FP4_ROWBYTES(128u) + i] = i < 64u ? (uint8_t)(s >> 56) : (uint8_t)127u;
            }
    };
    std::vector<uint8_t> qh((size_t)run_n * n_head * rb), ch((size_t)n_comp * rb);
    fill(qh, run_n * n_head);
    fill(ch, n_comp);   /* rows past vis: garbage the padded run must ignore */
    std::vector<float> wh((size_t)run_n * n_head, 0.25f);
    const uint64_t big = 2ull << 20;
    pulsar_gpu_tensor *q = pulsar_gpu_tensor_alloc(qh.size());
    pulsar_gpu_tensor *w = pulsar_gpu_tensor_alloc(wh.size() * 4u);
    pulsar_gpu_tensor *pad = pulsar_gpu_tensor_alloc(ch.size());
    pulsar_gpu_tensor *edge = pulsar_gpu_tensor_alloc(big);
    pulsar_gpu_tensor *sa = pulsar_gpu_tensor_alloc((uint64_t)run_n * n_comp * 4u);
    pulsar_gpu_tensor *sb = pulsar_gpu_tensor_alloc((uint64_t)run_n * n_comp * 4u);
    pulsar_gpu_tensor *flush = edge ? pulsar_gpu_tensor_view(edge, big - (uint64_t)vis * rb, (uint64_t)vis * rb) : NULL;
    std::vector<float> a((size_t)run_n * n_comp), b((size_t)run_n * n_comp);
    if (!q || !w || !pad || !edge || !sa || !sb || !flush || !pulsar_gpu_tensor_write(q, 0, qh.data(), qh.size()) ||
        !pulsar_gpu_tensor_write(w, 0, wh.data(), wh.size() * 4u) ||
        !pulsar_gpu_tensor_write(pad, 0, ch.data(), ch.size()) ||
        !pulsar_gpu_tensor_write(flush, 0, ch.data(), (uint64_t)vis * rb)) {
        fail("idx-decode-bound: alloc / upload");
    } else if (!pulsar_gpu_indexer_scores_decode_run_tensor(sa, q, w, pad, n_comp, run_n, run_pos0, n_head, 128u, ratio,
                                                            1.0f) ||
               !pulsar_gpu_indexer_scores_decode_run_tensor(sb, q, w, flush, n_comp, run_n, run_pos0, n_head, 128u,
                                                            ratio, 1.0f) ||
               !pulsar_gpu_synchronize() || !pulsar_gpu_tensor_read(sa, 0, a.data(), a.size() * 4u) ||
               !pulsar_gpu_tensor_read(sb, 0, b.data(), b.size() * 4u)) {
        fail("idx-decode-bound: launch (a fault here is the read past the bank's visible rows)");
    } else {
        int bad = 0;
        for (size_t i = 0; i < a.size(); i++) bad += memcmp(&a[i], &b[i], 4) != 0;
        printf("  5 idx-decode-bound: %u visible of %u, bank flush at the allocation end: %d scores differ\n", vis,
               n_comp, bad);
        if (bad) fail("idx-decode-bound: the scores depend on rows past the visible bound");
    }
    pulsar_gpu_tensor_free(flush);
    pulsar_gpu_tensor_free(q);
    pulsar_gpu_tensor_free(w);
    pulsar_gpu_tensor_free(pad);
    pulsar_gpu_tensor_free(edge);
    pulsar_gpu_tensor_free(sa);
    pulsar_gpu_tensor_free(sb);
}

/* ---- 6. ring-store-descr --------------------------------------------------------------------------------------- */

/// Case 6: the 0731 ring batch store with positions but no seq_id refuses (it dereferenced seq_id unchecked); with no
/// descriptor, and with a full one, it runs.
static void case_ring_store_descr(void) {
    const uint32_t hd = 512u, cap = 8u, n = 2u, banks = 2u;
    pulsar_gpu_tensor *ring = pulsar_gpu_tensor_alloc((uint64_t)banks * cap * PULSAR_ATTN_PACK_ROWBYTES(hd));
    pulsar_gpu_tensor *kv = pulsar_gpu_tensor_alloc((uint64_t)n * hd * 4u);
    pulsar_gpu_tensor *pos = pulsar_gpu_tensor_alloc(n * 4u);
    pulsar_gpu_tensor *seq = pulsar_gpu_tensor_alloc(n * 4u);
    const std::vector<float> x((size_t)n * hd, 0.5f);
    const int32_t p[2] = {3, 4}, q[2] = {0, 1};
    if (!ring || !kv || !pos || !seq || !pulsar_gpu_tensor_write(kv, 0, x.data(), x.size() * 4u) ||
        !pulsar_gpu_tensor_write(pos, 0, p, sizeof(p)) || !pulsar_gpu_tensor_write(seq, 0, q, sizeof(q))) {
        fail("ring-store-descr: alloc / upload");
    } else {
        const bool none = pulsar_gpu_attn_pack_ring_store_batch_tensor(ring, kv, cap, 0u, n, hd, NULL, NULL, 1u) != 0;
        const bool full = pulsar_gpu_attn_pack_ring_store_batch_tensor(ring, kv, cap, 0u, n, hd, pos, seq, banks) != 0;
        const bool half = pulsar_gpu_attn_pack_ring_store_batch_tensor(ring, kv, cap, 0u, n, hd, pos, NULL, banks) != 0;
        const bool synced = pulsar_gpu_synchronize() != 0;
        printf("  6 ring-store-descr: no descriptor %s, full %s, positions without seq_id %s, device %s\n",
               none ? "ran" : "REFUSED", full ? "ran" : "REFUSED", half ? "RAN" : "refused", synced ? "clean" : "FAULT");
        if (!none || !full || half || !synced) fail("ring-store-descr: the descriptor contract");
    }
    pulsar_gpu_tensor_free(ring);
    pulsar_gpu_tensor_free(kv);
    pulsar_gpu_tensor_free(pos);
    pulsar_gpu_tensor_free(seq);
}

/* ---- 7. hc-fused-width ----------------------------------------------------------------------------------------- */

/// Case 7: the fused HC split + norm runs n_embd 4096 and 5120 and refuses 4352 (between them: it used to launch the
/// 5120 unroll with dead groups -- the s51 corruption).  Weights from a synthetic host slab.
static void case_hc_fused_width(void) {
    const uint32_t n_hc = 4u, mix = 2u * n_hc + n_hc * n_hc, rows = 2u, wmax = 5120u;
    static float slab[3 + 24 + 5120];
    for (float &v : slab) v = 0.5f;
    const uint64_t off_scale = 0, off_base = 3u * 4u, off_norm = (3u + mix) * 4u;
    int bad = 0;
    for (uint32_t n_embd : {4096u, 4352u, 5120u}) {
        pulsar_gpu_tensor *norm_out = pulsar_gpu_tensor_alloc((uint64_t)rows * wmax * 4u);
        pulsar_gpu_tensor *split = pulsar_gpu_tensor_alloc((uint64_t)rows * mix * 4u);
        pulsar_gpu_tensor *pre = pulsar_gpu_tensor_alloc((uint64_t)rows * n_hc * 4u);
        pulsar_gpu_tensor *mixt = pulsar_gpu_tensor_alloc((uint64_t)rows * mix * 4u);
        pulsar_gpu_tensor *res = pulsar_gpu_tensor_alloc_elt((uint64_t)rows * n_hc * wmax, 2u, PULSAR_HC_ELT_FMT);
        if (!norm_out || !split || !pre || !mixt || !res) {
            fail("hc-fused-width: alloc");
        } else {
            const std::vector<uint8_t> z((size_t)rows * n_hc * wmax * 2u, 0u);
            pulsar_gpu_tensor_write(res, 0, z.data(), z.size());
            pulsar_gpu_tensor_write(mixt, 0, z.data(), (uint64_t)rows * mix * 4u);
            pulsar_gpu_tensor_write(pre, 0, z.data(), (uint64_t)rows * n_hc * 4u);
            const bool ran = pulsar_gpu_hc_split_weighted_sum_norm_f16_tensor(
                                 NULL, norm_out, NULL, NULL, 0, NULL, 0u, split, pre, mixt, res, slab, sizeof(slab),
                                 off_scale, off_base, off_norm, rows, n_embd, n_hc, 2u, 1e-6f, 1e-6f, 0) != 0 &&
                             pulsar_gpu_synchronize() != 0;
            if (ran != (n_embd != 4352u)) {
                fprintf(stderr, "  hc-fused-width: n_embd %u %s\n", n_embd, ran ? "ran" : "refused");
                bad++;
            }
        }
        pulsar_gpu_tensor_free(norm_out);
        pulsar_gpu_tensor_free(split);
        pulsar_gpu_tensor_free(pre);
        pulsar_gpu_tensor_free(mixt);
        pulsar_gpu_tensor_free(res);
    }
    printf("  7 hc-fused-width: 4096 and 5120 run, 4352 refuses: %s\n", bad ? "NO" : "yes");
    if (bad) fail("hc-fused-width: a width other than 4096 / 5120 ran, or one of them refused");
}

/* ---- 8. ple-order ---------------------------------------------------------------------------------------------- */

/// Case 8: the Qwen PLE launcher with a verify capture wider than its batch refuses BEFORE queuing anything: the
/// streams (the conv kernel's output) and the conv state come back byte-identical.
static void case_ple_order(void) {
    const int T = 2;
    const uint64_t big = 1ull << 20;
    pulsar_gpu_tensor *wt = pulsar_gpu_tensor_alloc(4 * big);   /* the four bf16 weights, 1 MiB apiece */
    pulsar_gpu_tensor *kv = pulsar_gpu_tensor_alloc(2 * big);   /* key, value */
    pulsar_gpu_tensor *st = pulsar_gpu_tensor_alloc((uint64_t)T * PULSAR_QWEN_HC_HIDDEN * 2u);
    pulsar_gpu_tensor *cs = pulsar_gpu_tensor_alloc((uint64_t)PULSAR_QWEN_PLE_STATE * PULSAR_QWEN_HC_HIDDEN * 4u);
    pulsar_gpu_tensor *sr = pulsar_gpu_tensor_alloc(4 * big);
    pulsar_gpu_tensor *ri = pulsar_gpu_tensor_alloc(64);
    const size_t wsb = pulsar_qwen_ple_workspace_bytes(T);
    pulsar_gpu_tensor *ws = pulsar_gpu_tensor_alloc(wsb);
    const int32_t idx[8] = {0, 0, /* row_seq */ 0, 1, /* row_j */ 0, /* seq_first */ 2, /* seq_rows */ 0 /* seq_bank */};
    if (!wt || !kv || !st || !cs || !sr || !ri || !ws || !pulsar_gpu_tensor_write(ri, 0, idx, sizeof(idx))) {
        fail("ple-order: alloc");
    } else {
        std::vector<uint8_t> pat_s((size_t)st->bytes), pat_c((size_t)cs->bytes), back;
        for (size_t i = 0; i < pat_s.size(); i++) pat_s[i] = (uint8_t)(i * 7u + 1u);
        for (size_t i = 0; i < pat_c.size(); i++) pat_c[i] = (uint8_t)(i * 13u + 5u);
        const std::vector<uint8_t> z(4 * big, 0u);
        pulsar_gpu_tensor_write(wt, 0, z.data(), z.size());
        pulsar_gpu_tensor_write(kv, 0, z.data(), 2 * big);
        pulsar_gpu_tensor_write(st, 0, pat_s.data(), pat_s.size());
        pulsar_gpu_tensor_write(cs, 0, pat_c.data(), pat_c.size());
        const char *W = (const char *)wt->ptr;
        const int32_t *R = (const int32_t *)ri->ptr;
        pulsar_qwen_ple_dev w{(const uint16_t *)W, (const uint16_t *)(W + big), (const uint16_t *)(W + 2 * big),
                              (const uint16_t *)(W + 3 * big)};
        pulsar_qwen_rows rows{};
        rows.row_seq = R;
        rows.row_j = R + 2;
        rows.seq_first = R + 4;
        rows.seq_rows = R + 5;
        rows.seq_bank = R + 6;
        rows.n_seq = 1;
        rows.state_rows = (float *)sr->ptr;
        rows.n_state_rows = T + 1;   /* wider than the batch: refused */
        const int rc = pulsar_qwen_ple_launch(&w, (const float *)kv->ptr, (const float *)((const char *)kv->ptr + big),
                                              (uint16_t *)st->ptr, T, &rows, (float *)cs->ptr, ws->ptr, wsb, 0);
        const bool synced = pulsar_gpu_synchronize() != 0;
        back.resize(pat_s.size());
        const bool s_same = pulsar_gpu_tensor_read(st, 0, back.data(), back.size()) && back == pat_s;
        back.resize(pat_c.size());
        const bool c_same = pulsar_gpu_tensor_read(cs, 0, back.data(), back.size()) && back == pat_c;
        printf("  8 ple-order: capture %d rows of %d -> rc %d, streams %s, conv state %s\n", rows.n_state_rows, T, rc,
               s_same ? "untouched" : "WRITTEN", c_same ? "untouched" : "WRITTEN");
        if (rc == 0 || !synced || !s_same || !c_same) fail("ple-order: the refusal queued work first");
    }
    pulsar_gpu_tensor_free(wt);
    pulsar_gpu_tensor_free(kv);
    pulsar_gpu_tensor_free(st);
    pulsar_gpu_tensor_free(cs);
    pulsar_gpu_tensor_free(sr);
    pulsar_gpu_tensor_free(ri);
    pulsar_gpu_tensor_free(ws);
}

/* ---- 9. hc-expand-sizes ---------------------------------------------------------------------------------------- */

/// Case 9: the split expands refuse an operand short of out_hc's rows (they checked none): a block_out, block_add,
/// residual or split one row short each refuses; full operands run.
static void case_hc_expand_sizes(void) {
    const uint32_t n_embd = 4096u, n_tok = 3u, n_hc = 4u, mix = 2u * n_hc + n_hc * n_hc;
    const uint64_t hc_row = (uint64_t)n_hc * n_embd * 2u, f_row = (uint64_t)n_embd * 4u, m_row = mix * 4u;
    pulsar_gpu_tensor *out = pulsar_gpu_tensor_alloc_elt((uint64_t)n_tok * n_hc * n_embd, 2u, PULSAR_HC_ELT_FMT);
    pulsar_gpu_tensor *res = pulsar_gpu_tensor_alloc_elt((uint64_t)n_tok * n_hc * n_embd, 2u, PULSAR_HC_ELT_FMT);
    pulsar_gpu_tensor *blk = pulsar_gpu_tensor_alloc(n_tok * f_row);
    pulsar_gpu_tensor *add = pulsar_gpu_tensor_alloc(n_tok * f_row);
    pulsar_gpu_tensor *split = pulsar_gpu_tensor_alloc(n_tok * m_row);
    if (!out || !res || !blk || !add || !split) {
        fail("hc-expand-sizes: alloc");
    } else {
        const std::vector<uint8_t> z((size_t)n_tok * hc_row, 0u);
        pulsar_gpu_tensor_write(res, 0, z.data(), n_tok * hc_row);
        pulsar_gpu_tensor_write(blk, 0, z.data(), n_tok * f_row);
        pulsar_gpu_tensor_write(add, 0, z.data(), n_tok * f_row);
        pulsar_gpu_tensor_write(split, 0, z.data(), n_tok * m_row);
        pulsar_gpu_tensor *sb = pulsar_gpu_tensor_view(blk, 0, (n_tok - 1u) * f_row);
        pulsar_gpu_tensor *sa = pulsar_gpu_tensor_view(add, 0, (n_tok - 1u) * f_row);
        pulsar_gpu_tensor *sr = pulsar_gpu_tensor_view(res, 0, (n_tok - 1u) * hc_row);
        pulsar_gpu_tensor *ss = pulsar_gpu_tensor_view(split, 0, (n_tok - 1u) * m_row);
        const bool full = pulsar_gpu_hc_expand_split_tensor(out, blk, res, split, n_embd, n_hc) &&
                          pulsar_gpu_hc_expand_add_split_tensor(out, blk, add, res, split, n_embd, n_hc);
        const bool any_short = pulsar_gpu_hc_expand_split_tensor(out, sb, res, split, n_embd, n_hc) ||
                               pulsar_gpu_hc_expand_split_tensor(out, blk, sr, split, n_embd, n_hc) ||
                               pulsar_gpu_hc_expand_split_tensor(out, blk, res, ss, n_embd, n_hc) ||
                               pulsar_gpu_hc_expand_add_split_tensor(out, blk, sa, res, split, n_embd, n_hc);
        const bool synced = pulsar_gpu_synchronize() != 0;
        printf("  9 hc-expand-sizes: full operands %s, a short operand %s, device %s\n", full ? "ran" : "REFUSED",
               any_short ? "RAN" : "refused", synced ? "clean" : "FAULT");
        if (!full || any_short || !synced) fail("hc-expand-sizes: the operand size checks");
        pulsar_gpu_tensor_free(sb);
        pulsar_gpu_tensor_free(sa);
        pulsar_gpu_tensor_free(sr);
        pulsar_gpu_tensor_free(ss);
    }
    pulsar_gpu_tensor_free(out);
    pulsar_gpu_tensor_free(res);
    pulsar_gpu_tensor_free(blk);
    pulsar_gpu_tensor_free(add);
    pulsar_gpu_tensor_free(split);
}

/* ---- 10. subview-dry ------------------------------------------------------------------------------------------- */

/// Case 10: pulsar_tensor_subview carries the base's dry flag, format and element size (dry was left unset -- stack
/// garbage).  The views are built over a stack poisoned with 0xff first.
static void case_subview_dry(void) {
    pulsar_gpu_tensor *real = pulsar_gpu_tensor_alloc_elt(64u, 2u, PULSAR_ELT_BF16);
    if (!real) {
        fail("subview-dry: alloc");
        return;
    }
    pulsar_gpu_tensor dry = *real;
    dry.dry = 1;
    int bad = 0;
    for (int k = 0; k < 2; k++) {
        volatile unsigned char poison[256];
        for (int i = 0; i < 256; i++) poison[i] = 0xffu;
        const pulsar_gpu_tensor *b = k ? &dry : real;
        const pulsar_gpu_tensor v = pulsar_tensor_subview(b, 16u, 32u);
        bad += v.dry != b->dry || v.fmt != b->fmt || v.esz != b->esz || v.owner != 0 || v.bytes != 32u ||
               (const char *)v.ptr != (const char *)b->ptr + 16;
    }
    printf("  10 subview-dry: views of a real and a dry tensor carry dry / fmt / esz: %s\n", bad ? "NO" : "yes");
    if (bad) fail("subview-dry: a view lost its base's dry flag, format or element size");
    pulsar_gpu_tensor_free(real);
}

/* ---- 11. bf16-nan ---------------------------------------------------------------------------------------------- */

/// The narrowing before L335, kept HERE only as the sweep's reference for every non-NaN input.
/// @param f the value
/// @return (u + 0x7fff + lsb) >> 16
__device__ static uint16_t old_bf16_rne(float f) {
    const uint32_t u = __float_as_uint(f);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

/// Every f32 bit pattern: [0] non-NaN inputs whose bits moved, [1] NaN inputs not narrowed to NaN, [2] NaN inputs the
/// old expression narrowed to a non-NaN.
/// @param cnt the three counters (zeroed)
__global__ static void bf16_sweep_kernel(unsigned long long *cnt) {
    unsigned long long moved = 0, lost = 0, old_lost = 0;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < (1ull << 32);
         i += (uint64_t)gridDim.x * blockDim.x) {
        const uint32_t u = (uint32_t)i;
        const float f = __uint_as_float(u);
        const uint16_t b = pulsar_f32_to_bf16_rne(f), o = old_bf16_rne(f);
        const bool nan_in = (u & 0x7fffffffu) > 0x7f800000u;
        const bool nan_b = (b & 0x7fffu) > 0x7f80u, nan_o = (o & 0x7fffu) > 0x7f80u;
        if (!nan_in) moved += b != o;
        else {
            lost += !nan_b;
            old_lost += !nan_o;
        }
    }
    atomicAdd(&cnt[0], moved);
    atomicAdd(&cnt[1], lost);
    atomicAdd(&cnt[2], old_lost);
}

/// Case 11: pulsar_f32_to_bf16_rne over all 2^32 patterns: every non-NaN keeps its old bits, every NaN stays NaN.
static void case_bf16_nan(void) {
    unsigned long long *d = nullptr, h[3] = {0, 0, 0};
    if (!ok(cudaMalloc(&d, sizeof(h)), "bf16-nan malloc") || !ok(cudaMemset(d, 0, sizeof(h)), "bf16-nan memset"))
        return;
    bf16_sweep_kernel<<<1024, 256>>>(d);
    if (ok(cudaDeviceSynchronize(), "bf16-nan sweep") && ok(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost),
                                                           "bf16-nan download")) {
        printf("  11 bf16-nan: 2^32 patterns: %llu non-NaN moved, %llu NaN not kept (the old expression lost %llu)\n",
               h[0], h[1], h[2]);
        if (h[0] || h[1]) fail("bf16-nan: a non-NaN changed bits, or a NaN narrowed to a number");
    }
    cudaFree(d);
}

/* ---- 12. swiglu-nan -------------------------------------------------------------------------------------------- */

/// The shared SwiGLU before L335 (fminf / fmaxf), kept HERE only as the sweep's reference for non-NaN inputs.
/// @param g gate
/// @param u up
/// @param wv routing weight
/// @param clamp clamp bound
/// @return the old expression's value
__device__ static float old_swiglu(float g, float u, float wv, float clamp) {
    if (clamp > 1.0e-6f) { g = fminf(g, clamp); u = fminf(fmaxf(u, -clamp), clamp); }
    const float s = g / (1.0f + expf(-g));
    return s * u * wv;
}

/// One thread a (g, u) pair from 2^26 hashed bit patterns (every class of f32: normals, subnormals, +-0, +-inf, NaN),
/// at three clamps: [0] non-NaN pairs whose bits moved, [1] pairs with a NaN operand not giving NaN, [2] the pairs the
/// old expression turned from a NaN operand into a number.
/// @param cnt the three counters (zeroed)
__global__ static void swiglu_sweep_kernel(unsigned long long *cnt) {
    unsigned long long moved = 0, lost = 0, old_lost = 0;
    const float clamps[3] = {0.0f, 7.0f, 10.0f};
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < (1ull << 26);
         i += (uint64_t)gridDim.x * blockDim.x) {
        uint64_t h = (i + 1u) * 0x9e3779b97f4a7c15ull;
        h ^= h >> 31;
        h *= 0xbf58476d1ce4e5b9ull;
        h ^= h >> 29;
        /* half the pairs from raw patterns, half near the clamps (where fminf and the if-chain could differ) */
        float g = __uint_as_float((uint32_t)h), u = __uint_as_float((uint32_t)(h >> 32));
        if (i & 1u) {
            g = ((int)((h >> 8) % 4001u) - 2000) * 0.0078125f;
            u = ((int)((h >> 40) % 4001u) - 2000) * 0.0078125f;
        }
        for (float c : clamps) {
            const float a = pulsar_swiglu_elem(g, u, 0.375f, c), b = old_swiglu(g, u, 0.375f, c);
            const bool nan_in = g != g || u != u;
            if (!nan_in) moved += __float_as_uint(a) != __float_as_uint(b) && !(a != a && b != b);
            else {
                lost += a == a;
                old_lost += b == b;
            }
        }
    }
    atomicAdd(&cnt[0], moved);
    atomicAdd(&cnt[1], lost);
    atomicAdd(&cnt[2], old_lost);
}

/// Case 12: the one SwiGLU keeps every non-NaN pair's old bits and turns no NaN operand into a number.
static void case_swiglu_nan(void) {
    unsigned long long *d = nullptr, h[3] = {0, 0, 0};
    if (!ok(cudaMalloc(&d, sizeof(h)), "swiglu-nan malloc") || !ok(cudaMemset(d, 0, sizeof(h)), "swiglu-nan memset"))
        return;
    swiglu_sweep_kernel<<<1024, 256>>>(d);
    if (ok(cudaDeviceSynchronize(), "swiglu-nan sweep") && ok(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost),
                                                             "swiglu-nan download")) {
        printf("  12 swiglu-nan: 2^26 pairs x 3 clamps: %llu non-NaN moved, %llu NaN operands not kept (the fminf form "
               "lost %llu)\n", h[0], h[1], h[2]);
        if (h[0] || h[1]) fail("swiglu-nan: a non-NaN pair changed bits, or a NaN operand became a number");
    }
    cudaFree(d);
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
    case_idx_decode_bound();
    case_ring_store_descr();
    case_hc_fused_width();
    case_ple_order();
    case_hc_expand_sizes();
    case_subview_dry();
    case_bf16_nan();
    case_swiglu_nan();
    puts(g_fail ? "KERNEL CONTRACT GATE FAIL" : "KERNEL CONTRACT GATE PASS");
    return g_fail ? 1 : 0;
}

// SPDX-License-Identifier: MIT
// ds4_mmq.h - public C ABI for ds4's quantized matmul kernels.
//
// All functions are extern "C" so ds4.c / ds4_cuda.cu can call them
// without C++ compilation. Functions return 0 on success and non-zero on
// failure (with stderr error message). Device pointers are caller-owned.
//
// The live surface is four entries: init, the should-use oracle, and the two
// aligned-SoA IQ2_XXS routed-MoE GEMMs (pair gate/up and single).  The wide
// ABI this header used to document -- raw Q2_K/Q8_0 dense and MoE entries,
// the whole mmvq-backed vector half, the fused Q2_K-down pipelines, the
// aligned decode matvecs, derepack, and the pool-stream hook -- was removed
// with its callers in L066; the 2026-08-22 types sweep removed the ~250 lines
// of orphaned documentation that had outlived those declarations here.

#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// One-time init. Sets the current CUDA device and triggers lazy population
// of the device-info singleton. Safe to call repeatedly.
//
//   device: CUDA device ordinal (0 for the primary GPU).
// Returns 0 on success.
int ds4_mmq_init(int device);

// Query whether ds4_mmq is willing to handle a given matmul. Returns
//   1 if mmq is faster than dequant+cublas for this shape on this device,
//   0 otherwise (the caller refuses; there is no second path).
//
// Wraps ggml_cuda_should_use_mmq. type_x uses ds4 quant codes which match
// ggml's enum:
//   8  = Q8_0
//   10 = Q2_K
//   16 = IQ2_XXS
//
//   ne11:      batch dimension (number of activation columns / tokens).
//   n_experts: 0 for dense matmul, >0 for MoE (e.g. 256 for V4 Flash).
int ds4_mmq_should_use(int type_x, int64_t ne11, int64_t n_experts);

// Routed-MoE contract shared by both entries below.  For each (token,
// slot-within-token's-top-k) pair the kernel computes:
//
//   out[col, row] = sum_k W[ids[token, slot], row, k] * X[token, k]
//
// where col = token * n_expert_used + slot, row in [0, M).  The caller owns
// the downstream sum-weighted-by-router-weights reduction across the
// n_expert_used dimension (moe_mmq_swiglu_fold / moe_sum in moe.cu).
//
// Layouts:
//   X_f32:   device pointer, [n_tokens, K] F32 row-major (K innermost).
//   ids:     device pointer, [n_tokens, n_expert_used] int32_t row-major.
//            ids[t*n_expert_used + s] is the expert id for token t's
//            s-th routing slot.  Values must be in [0, n_experts).
//   out_*:   caller-allocated, M * n_tokens * n_expert_used floats.
//            Column-major: out[col*M + row].
//
// Weights are the weight-server aligned-SoA IQ2_XXS artifact
// (--repack-iq2-aligned; byte-neutral repack of the raw expert stream).
// block_iq2_xxs is 66 bytes, so the raw stream is only 2-byte aligned and
// every 32-bit code word costs two 16-bit loads; the artifact splits the
// block scales out and 64B-aligns the code stream:
//
//   [ __half dq[nblk] ][ pad to 64B ][ uint2 qs[nblk * 8] ]
//
// where nblk = n_experts * M * (K / 256) and block linear order matches the
// raw tensor byte order (expert-major, then row, then block).
//
// Activation staging is E4M3 + ue8m0 (block_mx_act_mmq; ds4_act_block.cuh),
// built once per call -- there has been no q8_1 activation since the D2R
// E4M3 arm became the only arm.
//
// K must be a multiple of 256.  n_expert_used must be one of the values
// the vendored mm_ids_helper template specialises on: 2, 4, 6, 8, 16, 32
// (any other value takes the generic MMQ kernel).  For V4
// Flash, n_expert_used = 6.
//
// Output is NOT nonfinite-sanitized: the routed-MoE consumers
// (moe_mmq_swiglu / moe_sum with guard_nonfinite=1) sanitize at read.
// New callers must sanitize at consumption.
//
// Returns 0 on success, non-zero on validation or launch failure.

// Paired gate/up entry: computes gate AND up over the same activation in a
// single call, so the E4M3 activation staging (and the mm_ids_helper
// bookkeeping) happens once instead of twice.  Both weights must be the same
// quant type and shape (M, K, n_experts); out_a / out_b each have the single-
// entry output layout.  On error neither output is guaranteed valid.
int ds4_mmq_iq2_xxs_moe_pair_soa(
    const void    * Wa_soa,
    const void    * Wb_soa,
    const float   * X_f32,
    const int32_t * ids,
    float         * out_a,
    float         * out_b,
    int             M,
    int             K,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    cudaStream_t    stream,
    /* Optional pre-encoded activation: the E4M3 + ue8m0 the producing norm
     * already wrote for X_f32, with act_kbp blocks per row. Supplying it makes
     * the input staging a gather instead of a re-encode -- identical bytes,
     * 1 B/element read instead of 4. Pass NULL/NULL/0 to encode from X_f32. */
    const void    * act_q,
    const void    * act_sf,
    int             act_kbp);

/* pulsar (plan 41b): IQ2_XXS single-tensor MoE over the aligned-SoA artifact.
 * Upstream shipped a pair entry but no IQ2 SINGLE soa entry, which is what a
 * routed DOWN needs when the down tensor is IQ2 rather than Q2_K (our v5mx).
 * Same contract as the pair entry at one weight/one output. */
int ds4_mmq_iq2_xxs_moe_soa(
    const void    * W_soa,
    const float   * X_f32,
    const int32_t * ids,
    float         * out_f32,
    int             M,
    int             K,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    cudaStream_t    stream,
    /* L158 inc 5: the producer's E4M3 encoding of X_f32 (+ ue8m0 plane, pitch); required */
    const void    * act_q,
    const void    * act_sf,
    int             act_kbp);

#ifdef __cplusplus
} // extern "C"
#endif

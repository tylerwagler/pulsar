/* pulsar_tessera.h -- Tessera window-trellis projections, value (BF16, folded) family (L255).
 *
 * Made with Tessera by Robert Tand - https://github.com/RobTand/tessera
 *
 * The kernel is Tessera's fused window kernel, vendored verbatim in tessera_routed_fused_window.cuh (see
 * VENDOR-TESSERA.md); this is pulsar's torch-free host side of it.  Two launches:
 *
 *   dense   y[M, N] = x[M, K] . W^T for one Tessera Linear (E = 1), bf16 in, bf16 out, split-K below the SM
 *           count exactly as Tessera's ``dense_k_split`` chooses;
 *   moe     the routed expert stack: gate/up with the SwiGLU epilogue fused (mode 0), the weighted down
 *           projection into route-ordered rows (mode 2), and the fixed-order token sum -- Tessera's
 *           ``FusedRoutedWindowMoE.__call__`` with the value family's bf16 activations.
 *
 * Both are deterministic (no atomics on the data path) and graph-capturable (the work counters are zeroed
 * in-stream).  The weight planes are the ones Tessera prepares at load time (``prepare_grouped_window_gemm``
 * + ``words_by_expert`` / ``compose_table16`` / ``block_desc`` / ``run_pair``); the container builder emits
 * them, the engine never re-derives them.  Byte parity with Tessera's own build is tests/tessera_kernel_gate.
 */
#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** One projection stack of the value family, kernel-ready: E experts (1 for a dense Linear), each an
 *  N x K weight.  Every pointer is device memory. */
typedef struct {
    int E;                    /**< experts in the stack (1 for a dense Linear) */
    int K;                    /**< input columns: a multiple of 32, at least 128 */
    int N;                    /**< output rows: a multiple of 128 */
    const int32_t *words;     /**< [E, words_stride] packed window words, 16-byte aligned */
    long words_stride;        /**< int32 words per expert, a multiple of 4 */
    const uint16_t *table;    /**< [E, 16384] composed bf16 table (``compose_table16``) */
    const int32_t *init;      /**< [E, K] initial window state per column (permuted order) */
    const int32_t *has_init;  /**< [E] 1 when the expert carries initial states */
    const float *wscale;      /**< [E, N] row scales (folded into the bf16 weight before the dot) */
    const int32_t *runs;      /**< [E, 8] run pair (``run_pair``) */
    const int32_t *bdesc;     /**< [E, K / 32, 12] block descriptors (``block_desc``), 16-byte aligned */
    int tile_words;           /**< int32 words per 512-row tile: 16 * (sum of the column rates) */
    int slot_words;           /**< int32 words per word-stage slot (``slot_words_for_pair``) */
} pulsar_tessera_proj;

/** Tessera's ``dense_k_split``: how many ways the dense launch splits K at M rows on ``sms`` SMs. */
int pulsar_tessera_dense_k_split(int M, int N, int K, int sms, int tile_words);

/** Workspace the dense launch needs at M rows (work counter + the split-K fp32 partials). */
size_t pulsar_tessera_dense_workspace_bytes(const pulsar_tessera_proj *w, int M);

/** out[m, 0:N] = bf16(x[m, :] . W^T) for m < M.  ``x`` is bf16 [M, K] contiguous; ``out`` has unit column
 *  stride and row stride ``out_stride`` (even; a multiple of 4 when the launch splits K).  Returns 0, or -1
 *  after printing the refusal (a geometry the kernel does not serve, a short workspace, a CUDA error). */
int pulsar_tessera_dense_launch(const pulsar_tessera_proj *w, const uint16_t *x, int M, uint16_t *out,
                                long out_stride, void *ws, size_t ws_bytes, cudaStream_t stream);

/** Workspace the routed launch needs for T tokens at top_k (routing tables, activation and route rows). */
size_t pulsar_tessera_moe_workspace_bytes(const pulsar_tessera_proj *gate, const pulsar_tessera_proj *down,
                                          int T, int top_k);

/** out[t, :] = sum_j w[t, j] * down_e(silu(gate_e(x[t])) * up_e(x[t])), e = expert_ids[t, j], the sum in fixed
 *  j order in fp32 and rounded once.  ``x`` is bf16 [T, K]; ``expert_ids`` int32 [T, top_k] in [0, E);
 *  ``weights`` fp32 [T, top_k]; ``out`` bf16 [T, down->N].  gate and up share E, K and N; down is E x
 *  (K = gate->N).  Returns 0, or -1 after printing the refusal. */
int pulsar_tessera_moe_launch(const pulsar_tessera_proj *gate, const pulsar_tessera_proj *up,
                              const pulsar_tessera_proj *down, const uint16_t *x, int T, int top_k,
                              const int32_t *expert_ids, const float *weights, uint16_t *out, void *ws,
                              size_t ws_bytes, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

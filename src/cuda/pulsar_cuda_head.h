/** @file
 *  Model-head family options contract (pulsar_head_decl): the embedding norm and the logits finish. */
/* The model head's family options (L295 E5): what a family DECLARES about the
 * two ends of its stack, run by shared kernels.  The engine owns the head GEMM
 * (the bf16-rows front door, pulsar_rows_linear_launch), the sampler and the
 * logprobs; a family states only these facts (pulsar_head_decl):
 *
 *   embed norm   an RMSNorm of the token embedding row before the first layer
 *                (Inkling's `use_embed_norm`); NULL weight = none
 *   logit scale  a factor on every logit (Inkling's muP 1/16,
 *                `logits_mup_width_multiplier` 16); 1 = none
 *   real vocab   the head's rows past n_real are padding (Inkling: 201024
 *                rows, 200058 real); n_real == n_logits = none
 *
 * Embedding:  out[t] = bf16_rn((x * rstd) * w),  x = table[ids[t]] widened,
 *             rstd = rsqrtf(sum(x^2) / n_embd + eps), f32, one CTA a row with a
 *             fixed reduction order -- a row's bytes never depend on its batch.
 *             (vLLM's embed_rmsnorm: ONE rounding.  transformers rounds x * rstd
 *             to the dtype and multiplies by the weight in it: two.)
 *             Without a norm the row is copied as stored.
 * Logits:     the head GEMM writes f32 logits [rows][n_logits]; the finish
 *             pass makes, in place, logit[c] = logit[c] * scale for c < n_real
 *             and -inf for the padding.  The product is of the GEMM's FINAL f32
 *             value -- exactly what an alpha in the GEMM's f32 epilogue
 *             computes (vLLM folds 1/mup into addmm's alpha, beta 0) -- so moving
 *             the scale into a head arm's epilogue later is byte-identical.
 *             -inf is the padding mask every consumer already honours: the
 *             sampler, its argmax and its logprobs all skip a non-finite
 *             logit (pulsar_cuda_sampler.cu), so a pad id can never be
 *             sampled, ranked or scored. */
#ifndef PULSAR_CUDA_HEAD_H
#define PULSAR_CUDA_HEAD_H

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
/// The CUDA runtime's stream handle, declared for CUDA-free includers (same definition as cuda_runtime.h's).
typedef struct CUstream_st *cudaStream_t;
#endif

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** A family's head options. */
typedef struct {
    uint32_t        n_embd;          /**< the embedding / hidden width (a multiple of 8) */
    const uint16_t *embed_norm_w;    /**< device bf16 [n_embd], used as is (no 1 + w), or NULL: no embedding norm */
    /** the embedding norm's epsilon (>= 0; read only with embed_norm_w) */
    float           embed_norm_eps;
    uint32_t        n_logits;        /**< the head's rows = the logits row width (the padded vocab) */
    uint32_t        n_real;          /**< ids < n_real are real; 1 .. n_logits */
    float           logit_scale;     /**< finite, nonzero; 1 = none */
} pulsar_head_decl;

/** Token embedding rows: out[t] (bf16, at out + t * ld_out) = the declared
 *  norm of table[ids[t]] (bf16 [n_vocab][n_embd], 16-byte aligned).  `ids` is
 *  a device [n] array of ids the engine has already checked; an id outside
 *  [0, n_vocab) writes a NaN row (never another row).
 *  @return 0, -1 refused (named on stderr, nothing launched), -3 launch failure. */
/// @param h       the head declaration
/// @param table   device bf16 embedding table [n_vocab][n_embd]
/// @param n_vocab rows in the table
/// @param ids     device [n] token ids
/// @param n       rows to embed (one CTA each)
/// @param out     device bf16 output rows (16-byte aligned)
/// @param ld_out  out row pitch in elements (a multiple of 8, >= n_embd)
/// @param stream  stream to launch on
int pulsar_head_embed_launch(const pulsar_head_decl *h, const uint16_t *table, uint32_t n_vocab, const int32_t *ids,
                             uint32_t n, uint16_t *out, uint64_t ld_out, cudaStream_t stream);

/** The logits finish, in place on the head GEMM's f32 output: rows [n][ld]
 *  (ld >= n_logits), columns [0, n_logits).  Nothing launches when the head
 *  declares neither a scale nor padding.
 *  @return 0, -1 refused, -3 launch failure. */
/// @param h      the head declaration
/// @param logits device f32 logits rows, finished in place
/// @param n      rows (1 .. 65535)
/// @param ld     row pitch in floats (>= n_logits)
/// @param stream stream to launch on
int pulsar_head_logits_finish_launch(const pulsar_head_decl *h, float *logits, uint32_t n, uint64_t ld, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* PULSAR_CUDA_HEAD_H */

/* Qwen3.8-Flash-Next PLE n-gram ids and table (L251 S4).
 *
 * The source is transformers' Qwen4ExpTextNGramEmbedding: per token, the
 * bigram and the trigram ending at it, each hashed by 8 heads into its own
 * prime-sized range of one 320M-row table of 160 bf16 values:
 *
 *   s0 = x_t,  s1 = x_{t-1},  s2 = x_{t-2}   (shifted, EOS-reset: a shifted
 *        token is EOS when an EOS lies between it and t -- see below)
 *   m2 = s0*mult[0] ^ s1*mult[1],  m3 = m2 ^ s2*mult[2]
 *   id[h]     = m2 % prime[h]     + offset[h]       h = 0..7   (bigram heads)
 *   id[8 + h] = m3 % prime[8 + h] + offset[8 + h]   (trigram heads)
 *
 * The EOS reset, reduced from `_shift_right_ignore_eos`: the shift-k token is
 * valid only when no EOS sits in positions [t-k, t-1]; an invalid one reads as
 * EOS.  So s1 = x_{t-1} always (an EOS there IS the EOS it would read as) and
 * s2 = EOS when x_{t-1} is EOS, else x_{t-2}.  Positions before the sequence
 * are EOS.  That makes the whole per-sequence state the last two token ids
 * (the source keeps exactly that as a conv state), EOS EOS at the start.
 *
 * Integer arithmetic on non-negative values, as the Engram hash: every token
 * id is in [0, vocab) and every multiplier odd and below INT64_MAX / vocab (the
 * source's construction), so no product sets the sign bit, the XOR of
 * non-negatives is non-negative, and `%` is the source's torch.remainder.
 * An id outside [0, vocab) is refused -- a wrong row that is still in range is
 * the failure this must not have.
 *
 * The layout (multipliers, primes, offsets) is the checkpoint's own buffers
 * (`ple_embedding.layer_multipliers`, `ngram_heads_vocab_sizes`,
 * `ngram_heads_offsets`), carried by the artifact -- the engine never
 * recomputes them.  The rows are read with the Engram pool
 * (pulsar_engram_table_open_parts): 128 row-contiguous bf16 parts, 320 B per
 * row, 16 rows per token = the 2560-wide embedding in head order. */
#ifndef PULSAR_QWEN_NGRAM_H
#define PULSAR_QWEN_NGRAM_H

#include <stdint.h>

#define PULSAR_QWEN_NGRAM_ORDER    3u    /**< ngram_size: bigrams and trigrams */
#define PULSAR_QWEN_NGRAM_HEADS    8u    /**< heads_per_ngram */
#define PULSAR_QWEN_NGRAM_COLS     ((PULSAR_QWEN_NGRAM_ORDER - 1u) * PULSAR_QWEN_NGRAM_HEADS)   /**< 16 */
#define PULSAR_QWEN_NGRAM_ROW_BYTES 320u /**< 160 bf16 */

typedef struct {
    int64_t  mult[PULSAR_QWEN_NGRAM_ORDER];    /**< layer_multipliers (odd) */
    uint64_t prime[PULSAR_QWEN_NGRAM_COLS];    /**< ngram_heads_vocab_sizes */
    uint64_t offset[PULSAR_QWEN_NGRAM_COLS];   /**< ngram_heads_offsets */
    int32_t  eos;                              /**< eos_token_id (the first, if a list) */
    uint32_t vocab;                            /**< vocab_size: every token id is below it */
    uint64_t n_rows;                           /**< rows in the table (the padded vocab) */
} pulsar_qwen_ngram_layout;

/** The per-sequence state: the last two token ids, oldest first. */
typedef struct {
    int32_t prev[2];
} pulsar_qwen_ngram_ctx;

/** Checks the layout's invariants (odd multipliers below INT64_MAX / vocab,
 *  every head range inside the table, EOS a token).  1 = usable; 0 with the
 *  broken invariant printed. */
int  pulsar_qwen_ngram_layout_check(const pulsar_qwen_ngram_layout *L);

/** A new sequence: EOS, EOS. */
void pulsar_qwen_ngram_ctx_init(const pulsar_qwen_ngram_layout *L, pulsar_qwen_ngram_ctx *ctx);

/** The table rows of `n` tokens that follow `ctx`: rows[i * 16 + h], head
 *  order; advances ctx past them.  Refuses (dies) on a token outside the
 *  vocab. */
void pulsar_qwen_ngram_rows(const pulsar_qwen_ngram_layout *L, pulsar_qwen_ngram_ctx *ctx,
                            const int32_t *ids, uint32_t n, uint64_t *rows);

#endif /* PULSAR_QWEN_NGRAM_H */

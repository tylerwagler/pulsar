/* CandidateSelector chain head (L299) -- a block drafter's last stage, family-neutral: the second chain-head kind of
 * the block lane (spec_block.h), beside the LiLiCorr candidate head (pulsar_cuda_lilicorr.h).
 *
 * A block drafter drafts the block's mask slots in one forward; its LM head gives each slot's top-K candidates.  The
 * selector (z-lab DFlash2's CandidateSelector, dflash/model.py @07ebd93 -- the DFlash2 drafters for Qwen3.8 carry it)
 * re-scores each slot's candidates against the token chosen at the slot before, and a left-to-right walk picks ONE
 * chain.  The reference is tools/dflash2/replay.py (Drafter.propose, == z-lab's propose to 1e-5).
 *
 * THE HEAD, per bank (SL = block - 1 slots, K candidates a slot, a rank-R factorisation over the vocabulary):
 *   hp[s]         = hidden_projection(pass[s])               the drafter's final-norm row of slot s, R wide (the rows
 *                                                            front door's Linear: the caller runs it, bf16 rows)
 *   score(s, p, c) = unary[s][c] + sum_r succ[cand[s][c]][r] * (pred[p][r] * hp[s][r])
 *                   p the token chosen at slot s - 1 (the bank's carry -- the anchor -- for slot 0); succ / pred the
 *                   two [vocab][R] codebooks; unary the candidate's score from the drafter's head
 *   scores[s][i][c] = score(s, cand[s - 1][i], c)   for s > 0 (every predecessor candidate i: a sampled walk may take
 *                    any of them); score(s = 0, anchor, c) in all K rows of slot 0
 *   walk          prev = 0; for s: j = FIRST argmax of scores[s][prev][:]; chain_rows[s] = j; chain_ids[s] =
 *                 cand[s][j]; prev = j                                    (torch.argmax: ties to the lower index)
 * The unary the lane feeds is each candidate's log-softmax over the head's row: z-lab adds the raw logit, which is
 * the same plus a per-row constant -- no walk and no candidate distribution moves (both are invariant to a shift of a
 * slot's row).
 *
 * NUMERICS: every operand widened to f32; pred[p][r] * hp[s][r] rounded once to f32, the R products with succ summed
 * in r order with fused multiply-adds, the unary added last.  Deterministic, and a bank's bytes never depend on the
 * banks beside it.
 *
 * LIBRARIES: the hidden_projection is the caller's (the rows front door); the codebook gathers, the dots and the walk
 * are this one kernel -- nothing a library covers at K = 16, R = 256.  No host sync, no allocation, no cuBLASLt:
 * capture-safe (the lane runs it inside the draft pass's graph). */
#ifndef PULSAR_CUDA_CANDSEL_H
#define PULSAR_CUDA_CANDSEL_H

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
typedef struct CUstream_st *cudaStream_t;
#endif

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PULSAR_CANDSEL_TOPK_MAX 16u   /**< candidates a slot (the block lane's top-k cap) */
#define PULSAR_CANDSEL_RANK_MAX 256u  /**< the codebooks' width (the shared-memory rows: k x rank x 6 B <= 24 KiB) */
#define PULSAR_CANDSEL_SLOTS_MAX 15u  /**< slots a block (the lane's widest block, 16, minus the carry) */

/** The selector over n_banks banks.  Device inputs:
 *    cand    [n_banks][slots][k] i32    vocabulary ids, each in [0, vocab)
 *    unary   [n_banks][slots][k] f32    their scores from the drafter's head
 *    hp      bank b's slot s at hp + (b * hp_bank + s) * hp_ld, bf16 [rank]: the projected pass rows
 *    anchor  [n_banks] i32              each bank's carry (slot 0's predecessor)
 *    pred, succ [vocab][rank] bf16      the predecessor / successor codebooks
 *  Device outputs: scores [n_banks][slots][k][k] f32, chain_ids / chain_rows [n_banks][slots] i32 (the header).
 *  1 <= k <= PULSAR_CANDSEL_TOPK_MAX, 1 <= rank <= PULSAR_CANDSEL_RANK_MAX, 1 <= slots <= PULSAR_CANDSEL_SLOTS_MAX.
 *  @return 0, -1 refused (named on stderr, nothing launched), -3 launch failure. */
int pulsar_candsel_launch(uint32_t n_banks, uint32_t slots, uint32_t k, uint32_t rank, uint32_t vocab,
                          const int32_t *cand, const float *unary, const uint16_t *hp, uint64_t hp_bank,
                          uint64_t hp_ld, const int32_t *anchor, const uint16_t *pred, const uint16_t *succ,
                          float *scores, int32_t *chain_ids, int32_t *chain_rows, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* PULSAR_CUDA_CANDSEL_H */

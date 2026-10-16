/* LiLiCorr candidate-lattice head + path selector (L298) -- the block drafter's last stage, family-neutral.
 *
 * A block drafter (DFlash-style: one pass drafts a block of 1 anchor + 7 mask rows) ends in a candidate head: the top 8
 * tokens of its own LM head at each of the 7 drafted slots (56 candidates) are re-scored jointly by NVIDIA's LiLiCorr
 * correlator, and a greedy walk over the transition scores picks ONE 7-token chain.  The reference is vLLM d9a543c5
 * models/lilicorr.py (LiLiCorrHead.score / forward) and v1/worker/gpu/spec_decode/dflash2/speculator.py
 * (_selector_walk_kernel at temperature 0); tools/lilicorr/reference.py restates both in PyTorch and its fixtures grade
 * this kernel (tests/lilicorr_head_gate.cu).
 *
 * THE HEAD, per bank (H = 1024, 8 heads x 128, the lattice n = 7 slots x 8 candidates = 56 rows, slot-major):
 *   x[s,k]  = token_proj(E[cand[s,k]]) + pass_hidden_proj(pass[s]) + feature_mlp(LN5(f[s,k])) + slot_emb[s] + rank_emb[k]
 *             f = (lp, exp(lp), lp - max_k lp, k / 7, k == 0), lp = the candidate's log-prob; E = the TARGET's input
 *             embedding; pass[s] = the drafter's final-norm output at block row s + 1
 *   a       = context_proj(anchor_hidden) * anchor_valid       anchor_hidden = hidden_norm(fc(taps at the last kept row))
 *   2 x     { x += out_proj(MHA(attn_norm(x), bias = rel_slot_bias[h][slot(q) - slot(k) + 7] + same_slot_bias[h] *
 *             [slot(q) == slot(k)]));  x += down(silu(up(mlp_norm(x)))) }      the 56-row lattice, no causal mask
 *   x       = output_norm(x);  a = anchor_norm(a)
 *   g       = silu(factor_input_proj [x | a | x * a])       (vLLM splits the [1024][3072] weight into three GEMMs
 *             and rounds each partial to bf16; here ONE GEMM over the concatenated row, f32 throughout -- closer to
 *             the exact sum, inside the head's bf16 noise floor)
 *   out, in = normalize(out_head g), normalize(in_head g)      (L2, eps 1e-4)
 *   start[j] = normalize(anchor_out_head a) . in[0,j];   pair[s][i][j] = out[s,i] . in[s+1,j]
 *   scores  = 8 * [ start broadcast over the 8 rows of slot 0 ; pair ]               [7][8][8] f32
 *   walk    prev = 0; for s: j = FIRST argmax of scores[s][prev][:]; chain_rows[s] = j; chain_ids[s] = cand[s][j];
 *           prev = j                                                  (triton tl.max(return_indices): ties to the left)
 *
 * NUMERICS: bf16 activations rounded where vLLM's torch ops round them (each Linear's output once -- f32 accumulate,
 * bias and the NVFP4 global scale in f32, one bf16 rounding; each residual add; each norm), f32 accumulation
 * everywhere.  The scores are graded against vLLM's within rounding, the chain exactly.
 *
 * LIBRARIES: every Linear is the BF16 rows arm's cuBLASLt GEMM (pulsar_bf16_rows_linear_launch, prompt = the
 * fixed-order kernel per shape); the lattice attention's Q K^T and P V are strided-batched cuBLASLt GEMMs
 * (pulsar_gpu_bf16_batched_gemm, no workspace).  The glue no library covers is this TU's: the candidate gather and
 * features, the epilogues (bias, global scale, residual, norm, activation, head-major split), the biased softmax, the
 * normalize-and-dot scores and the walk.  No host sync, no allocation: graphable.
 *
 * WEIGHTS: bf16 device pointers, 16-byte aligned, row-major [out][in] as the checkpoint stores them.  An NVFP4 Linear
 * (token_proj, pass_hidden_proj, context_proj, the layers' out_proj / mlp up / mlp down) is dequantized to bf16 at load
 * as e2m1 x e4m3 (exact in bf16; pulsar_nvfp4_dequant_bf16_launch) and its f32 weight_scale_2 rides in `s2` (applied
 * in the epilogue).  A BF16 Linear has s2 = 1.  Every field binds one container role of the head block `lilicorr`
 * (the HF suffix names; layer -1 the head-level tensors, lattice layer i lilicorr.layers.i) as stored -- nothing is
 * re-laid at load. */
#ifndef PULSAR_CUDA_LILICORR_H
#define PULSAR_CUDA_LILICORR_H

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
typedef struct CUstream_st *cudaStream_t;
#endif

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PULSAR_LILICORR_SLOTS 7u       /**< drafted slots per block (block 8 = anchor + 7) */
#define PULSAR_LILICORR_TOPK 8u        /**< candidates per slot */
#define PULSAR_LILICORR_HIDDEN 1024u   /**< the head's width */
#define PULSAR_LILICORR_HEADS 8u       /**< lattice attention heads (x 128) */
#define PULSAR_LILICORR_MODEL 4096u    /**< the drafter's / target's hidden width (token, pass and anchor inputs) */
#define PULSAR_LILICORR_LAYERS 2u
#define PULSAR_LILICORR_BANKS_MAX 16u  /**< banks one launch takes */

/** One Linear: bf16 weight [out][in], optional bf16 bias [out], f32 global scale (1 for a BF16 weight). */
typedef struct {
    const uint16_t *w;
    const uint16_t *b;
    float s2;
} pulsar_lilicorr_linear;

typedef struct {
    const uint16_t *lat_attn_norm;         /**< [1024] */
    pulsar_lilicorr_linear lat_attn_in;    /**< in_proj: 1024 -> 3072 (q, k, v rows), BF16 */
    pulsar_lilicorr_linear lat_attn_out;   /**< out_proj: 1024 -> 1024, NVFP4 */
    const uint16_t *lat_mlp_norm;          /**< [1024] */
    pulsar_lilicorr_linear lat_mlp_up;     /**< 1024 -> 2048, NVFP4 */
    pulsar_lilicorr_linear lat_mlp_down;   /**< 2048 -> 1024, NVFP4 */
} pulsar_lilicorr_layer;

/** The head's weights (checkpoint prefix `lilicorr.`). */
typedef struct {
    pulsar_lilicorr_linear token_proj;     /**< 4096 -> 1024, NVFP4 */
    pulsar_lilicorr_linear pass_hidden_proj; /**< 4096 -> 1024, NVFP4 */
    pulsar_lilicorr_linear context_proj;   /**< 4096 -> 1024, NVFP4 */
    const uint16_t *feature_norm_w;        /**< [5] LayerNorm, eps 1e-5 (torch's default: vLLM builds
                                                nn.LayerNorm(5) with no eps and the config carries none) */
    const uint16_t *feature_norm_b;        /**< [5] */
    const uint16_t *feature_mlp_up_w;      /**< [1024][5] bf16 (K = 5: the glue's, no GEMM) */
    const uint16_t *feature_mlp_up_b;      /**< [1024] */
    pulsar_lilicorr_linear feature_mlp_down; /**< 1024 -> 1024, BF16 */
    const uint16_t *slot_embedding;        /**< [7][1024] */
    const uint16_t *rank_embedding;        /**< [8][1024] */
    const uint16_t *relative_slot_bias;    /**< [8 heads][15] */
    const uint16_t *same_slot_bias;        /**< [8 heads] */
    pulsar_lilicorr_layer layers[PULSAR_LILICORR_LAYERS];
    const uint16_t *output_norm;           /**< [1024] */
    const uint16_t *anchor_norm;           /**< [1024] */
    pulsar_lilicorr_linear factor_input_proj; /**< [x | a | x * a] 3072 -> 1024, BF16 */
    pulsar_lilicorr_linear out_head;       /**< 1024 -> 1024, BF16 */
    pulsar_lilicorr_linear in_head;        /**< 1024 -> 1024, BF16 */
    pulsar_lilicorr_linear anchor_out_head; /**< 1024 -> 1024, BF16 */
    float vector_eps;                      /**< lilicorr_vector_eps (1e-4) */
    float logit_scale;                     /**< lilicorr_logit_scale (8) */
} pulsar_lilicorr_head;

/** Device workspace bytes of one launch over n_banks banks (0 when n_banks is 0 or past PULSAR_LILICORR_BANKS_MAX). */
uint64_t pulsar_lilicorr_head_ws_bytes(uint32_t n_banks);

/** The head and the greedy walk for n_banks banks (1 .. PULSAR_LILICORR_BANKS_MAX).  Device inputs:
 *    cand_ids      [n_banks][7][8] i32   top-8 per slot, descending by log-prob (the draft LM head's order)
 *    cand_logprob  [n_banks][7][8] f32   log_softmax over the full vocabulary of those ids
 *    embed_table   [vocab][4096] bf16    the TARGET's input embedding; the head gathers its 56 rows a bank
 *    pass_hidden   [n_banks][7][4096] bf16  the drafter's final-norm output at block rows 1..7
 *    anchor_hidden [n_banks][4096] bf16  hidden_norm(fc(taps at the last committed position))
 *    anchor_valid  [n_banks] u8          0: the step kept no row (the anchor term is zeroed)
 *  Device outputs:
 *    scores        [n_banks][7][8][8] f32  forward()'s logits (slot 0: the start scores in all 8 rows)
 *    chain_ids     [n_banks][7] i32      the walk's tokens
 *    chain_rows    [n_banks][7] i32      the candidate index chosen per slot (scores[s][chain_rows[s-1]] is slot s's
 *                                        realized row, row 0 for s = 0)
 *  ws: pulsar_lilicorr_head_ws_bytes(n_banks) device bytes, 256-byte aligned.
 *  @return 0, -1 refused (named on stderr, nothing launched), -3 a launch failed. */
int pulsar_lilicorr_head_launch(const pulsar_lilicorr_head *w, uint32_t n_banks, const int32_t *cand_ids,
                                const float *cand_logprob, const uint16_t *embed_table, const uint16_t *pass_hidden,
                                const uint16_t *anchor_hidden, const uint8_t *anchor_valid, float *scores,
                                int32_t *chain_ids, int32_t *chain_rows, void *ws, uint64_t ws_bytes,
                                cudaStream_t stream);

/** ModelOpt NVFP4 -> bf16 at load: packed [rows][cols / 2] u8 (element 2i in the low nibble), scale [rows][cols / 16]
 *  E4M3 -> out [rows][cols] bf16 = e2m1 x e4m3 (exact; the f32 weight_scale_2 is NOT applied -- it is the Linear's
 *  s2).  cols % 16 == 0.  @return 0, -1 refused, -3 launch failure. */
int pulsar_nvfp4_dequant_bf16_launch(const uint8_t *packed, const uint8_t *scale_e4m3, uint32_t rows, uint32_t cols,
                                     uint16_t *out, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* PULSAR_CUDA_LILICORR_H */

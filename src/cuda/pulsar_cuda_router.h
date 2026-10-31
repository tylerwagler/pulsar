/** @file
 *  The MoE router contract: the family's declaration (pulsar_router_cfg) and the layer's tensors. */
/* The MoE router -- ONE parameterized implementation for every family (L295 F2).
 *
 * A family DECLARES its router (pulsar_router_cfg: the score function, the renormalisation, the route scale, how
 * its shared experts are gated) and binds its layer's tensors (pulsar_router_layer: a selection bias, an image-slot
 * bias, a token-id table, a global scale); the engine owns the kernels.  Two launches:
 *
 *   logits   pulsar_router_logits_launch: bf16 rows x bf16 weight rows -> f32 logits, one fixed-order
 *            accumulation per (row, logit) whatever the batch width.  A family whose router weight takes another
 *            format computes its logits through the format front door (linear.cpp) instead.
 *   select   pulsar_router_select_launch, one warp per row:
 *              score     s_e = f(l_e) over the routed logits l_0..l_{E-1} (bf16-rounded first when the family's
 *                        decision is made on bf16 logits): softmax | sqrt(softplus) | sigmoid;
 *              pick      the top-k of s_e + bias_e (the selection bias, or the image-slot bias on a row whose token
 *                        id says it is an image slot) in the total order (descending, ties to the LOWER expert id,
 *                        torch.topk's order; a row whose s_e + bias_e is not all finite has no routing); or, on a
 *                        token-id table layer, the table's row for the token;
 *              weights   SUM: w_k = s_k / max(sum_k s_k, floor) * route_scale, the picks' UNBIASED scores;
 *                        LOGSIGMOID_SINKS: over the k picks' RAW logits and the S sink logits (rows E..E+S-1),
 *                        w_i = softmax_i(logsigmoid(a_i)) * route_scale * global_scale -- the last S of them gate
 *                        the S shared experts;
 *              shared    NONE: no gate rows (an ungated shared expert, scale 1); SIGMOID: gamma_j = sigmoid(l_{E+j});
 *                        SINK: gamma_j = the renormalisation's sink weight j.
 *
 * Rows of one launch never interact: a row's picks and weights are the same bytes at any batch width.  A row whose
 * scores are not all finite has no routing; it is routed to experts 0..k-1 with NaN weights and NaN gammas, so the
 * NaN reaches the block output and the MoE sum's non-finite flag names the layer (rule 9). */
#ifndef PULSAR_CUDA_ROUTER_H
#define PULSAR_CUDA_ROUTER_H

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
/// The CUDA runtime's stream handle, declared for CUDA-free includers (same definition as cuda_runtime.h's).
typedef struct CUstream_st *cudaStream_t;
#endif
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** The most shared experts a router gates (and the rows MoE front door carries). */
#define PULSAR_ROUTER_SHARED_MAX 4

/** The score function over the routed logits. */
typedef enum {
    PULSAR_ROUTER_SCORE_SOFTMAX = 1,        /**< softmax over the E routed logits (Qwen) */
    PULSAR_ROUTER_SCORE_SQRT_SOFTPLUS = 2,  /**< sqrt(softplus(l)) (DeepSeek V4) */
    PULSAR_ROUTER_SCORE_SIGMOID = 3,        /**< sigmoid(l) (Inkling) */
} pulsar_router_score;

/** How the picks' weights are normalised. */
typedef enum {
    PULSAR_ROUTER_RENORM_SUM = 1,               /**< w_k = s_k / max(sum s, floor) * route_scale */
    PULSAR_ROUTER_RENORM_LOGSIGMOID_SINKS = 2,  /**< softmax of logsigmoid over the k raw logits + S sink logits */
} pulsar_router_renorm;

/** How the shared experts are gated. */
typedef enum {
    PULSAR_ROUTER_SHARED_NONE = 0,     /**< no gate rows: each shared expert's output is added at scale 1 */
    PULSAR_ROUTER_SHARED_SIGMOID = 1,  /**< logit row E + j, sigmoid: gamma_j (Qwen's shared_expert_gate) */
    PULSAR_ROUTER_SHARED_SINK = 2,     /**< logit row E + j joins the renormalisation (needs LOGSIGMOID_SINKS) */
} pulsar_router_shared_gate;

/** What a family declares about its router.  Plain values: one per model (per drafter), not per layer. */
typedef struct pulsar_router_cfg {
    int n_expert;           /**< routed experts E: logit rows 0..E-1 */
    int top_k;              /**< picks per row */
    int n_shared;           /**< shared experts S (0..PULSAR_ROUTER_SHARED_MAX) */
    /** how the shared experts are gated */
    pulsar_router_shared_gate shared_gate;
    /** the routed experts' score function */
    pulsar_router_score score;
    /** how the picked scores become weights */
    pulsar_router_renorm renorm;
    bool logits_bf16;       /**< round each logit to bf16 before scoring (the decision is made on bf16 logits) */
    bool weights_bf16;      /**< round each output weight and gamma to bf16 */
    float sum_floor;        /**< RENORM_SUM: the denominator's floor */
    float route_scale;      /**< multiplies every routed weight (and, under the sink renormalisation, the sink
                             *  gammas; a SIGMOID gamma is the gate row's sigmoid alone) */
} pulsar_router_cfg;

/** One layer's router tensors (device pointers; any may be NULL). */
typedef struct pulsar_router_layer {
    const float *bias;            /**< [E] selection bias: picks by s_e + bias_e; weights stay unbiased */
    const float *alt_bias;        /**< [E] the bias a row takes INSTEAD when its token id >= alt_token_min (an image
                                   *  slot); such a row never takes the table */
    /** a row whose token id is >= this takes alt_bias (read only when alt_bias is set) */
    int32_t alt_token_min;
    const int32_t *table;         /**< [table_rows][top_k] expert ids by token id (a hash-routed layer): replaces the
                                   *  top-k; an id outside [0, table_rows) reads row 0; weights from the scores */
    /** rows in `table` (> 0 when a table is bound) */
    int32_t table_rows;
    const float *global_scale;    /**< [1] multiplies the sink renormalisation's weights, or NULL */
} pulsar_router_layer;

/** Logit rows a router reads per token: E, plus S when its shared experts are gated by logit rows. */
/// @param c the router declaration (NULL: 0)
/// @return E + S when the shared experts are logit-gated, else E
int pulsar_router_logit_rows(const pulsar_router_cfg *c);

/** logits [T][n_w + n_tail] f32 = x [T][hidden] (bf16) against bf16 weight rows w [n_w][hidden] then w_tail
 *  [n_tail][hidden] (NULL when n_tail is 0): f32 accumulation in one fixed order per (row, logit), so a row's logits
 *  do not depend on the rows beside it.  hidden % 8 == 0 and 16-byte aligned rows.  0 on success. */
/// @param x_bf16 device bf16 activations [T][hidden]
/// @param T      token rows
/// @param hidden hidden width (a multiple of 8)
/// @param w      device bf16 routed-expert rows [n_w][hidden]
/// @param n_w    rows in w (> 0)
/// @param w_tail device bf16 trailing rows [n_tail][hidden] (the shared-expert gates), or NULL
/// @param n_tail rows in w_tail (>= 0; non-zero exactly when w_tail is set)
/// @param logits device f32 out [T][n_w + n_tail]
/// @param stream stream to launch on
/// @return 0, -1 refused (named on stderr), -3 launch failure
int pulsar_router_logits_launch(const uint16_t *x_bf16, int T, int hidden, const uint16_t *w, int n_w,
                                const uint16_t *w_tail, int n_tail, float *logits, cudaStream_t stream);

/** Route T rows from their logits [T][pulsar_router_logit_rows(c)]: selected [T][top_k] (pick order: descending
 *  selection score, or the table's order), weights [T][top_k], gammas [S][T] (one plane per gated shared expert;
 *  NULL when S == 0 or the gate is NONE), probs [T][E] = the unbiased scores (optional).  `tokens` [T] (device) is
 *  read by a layer with an alt_bias or a table (a table needs them).  Refuses (-1) a config or a shape without an
 *  instance; 0 on success. */
/// @param c        the family's router declaration
/// @param l        the layer's router tensors
/// @param logits   device f32 [T][pulsar_router_logit_rows(c)]
/// @param tokens   device [T] token ids, or NULL (needed for a table)
/// @param T        rows
/// @param selected device out [T][top_k] expert ids
/// @param weights  device out [T][top_k] routed weights
/// @param gammas   device out [S][T] shared-expert gates, or NULL when S == 0
/// @param probs    device out [T][E] unbiased scores, or NULL
/// @param stream   stream to launch on
/// @return 0, -1 refused (named on stderr), -3 launch failure
int pulsar_router_select_launch(const pulsar_router_cfg *c, const pulsar_router_layer *l, const float *logits,
                                const int32_t *tokens, int T, int32_t *selected, float *weights, float *gammas,
                                float *probs, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* PULSAR_CUDA_ROUTER_H */

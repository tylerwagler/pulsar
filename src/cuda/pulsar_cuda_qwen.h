/** @file
 *  Qwen3.8-Flash-Next block pieces contract: gated residual, PLE, embedding and the MTP combine (the rows front door
 *  they sit on: pulsar_cuda_rows.h). */
/* Qwen3.8-Flash-Next (HF `qwen4_exp`) block pieces -- L251 stream S4.
 *
 * The MoE block (the family's router declaration for the shared router of
 * pulsar_cuda_router.h -- softmax top-10, a sigmoid-gated shared expert -- and
 * the rows MoE front door's routed experts and N shared experts), the Gated Residual (GR: the 4-stream low-rank read, the
 * per-stream scalar write, and the collapse-only top-level mixer) and the PLE
 * n-gram injection at layer index 1.  The reference for every one of them is
 * transformers' modeling_qwen4_exp.py (Qwen4ExpTextSparseMoeBlock,
 * Qwen4ExpTextTopKRouter, Qwen4ExpTextGatedResidual, Qwen4ExpTextPLELayer).
 *
 * These are self-contained launchers over raw device pointers: no engine
 * tensor registry, no activation-cache globals.  The family interface (L251
 * S1) wires them into a layer plan; until then they are exercised by their
 * gates only (tests/qwen_*_gate.cu) and are parked with the branch.
 *
 * Numerics, and where they mirror the source.  The source runs bf16
 * end to end.  Here, as in the DeepSeek lane (pulsar_hc_t):
 *   - the residual STREAMS are bf16 storage, f32 math; every write rounds once;
 *   - the block input (the GR read's collapsed row) is rounded to bf16, like
 *     the source's `mixed_input` and DeepSeek's hc_pre row -- the router and
 *     the shared-expert gate (bf16 weights) read that row, and the A8
 *     consumers (EXL3 / MXFP8) read ITS E4M3 encoding (one value, two
 *     encodings, both emitted by the producer);
 *   - norm outputs, low-rank intermediates, gates and states are f32;
 *   - the router's logits are rounded to bf16 before the softmax and the
 *     top-10 weights after the renormalisation, exactly where F.linear /
 *     `.to(router_logits.dtype)` round them in the source: the routing DECISION
 *     is the model's, and it is made on bf16 logits.
 * Every activation that feeds an E4M3-weight or trellis GEMV is E4M3 with one
 * ue8m0 per 32 (the A8 contract), encoded by the kernel that produced it.
 *
 * Every kernel here is M-neutral: a row's arithmetic does not depend on how
 * many rows ride with it, so decode rows and prefill rows are one arithmetic.
 */
#ifndef PULSAR_CUDA_QWEN_H
#define PULSAR_CUDA_QWEN_H

#include "pulsar_cuda_rows.h"   /* L295 F8 Q5: the family-neutral rows front door these pieces sit on */

#ifdef __cplusplus
extern "C" {
#endif
/* ---- the shapes these kernels are built for (config.json of the checkpoint).
 * The launchers take the dims they are handed and REFUSE any that differ: the
 * config is the authority, these constants are what the templates were
 * instantiated for. */
#define PULSAR_QWEN_HIDDEN      2560   /**< hidden_size */
#define PULSAR_QWEN_HC          4      /**< hc_count: residual streams */
/** the four streams' total width (10240) */
#define PULSAR_QWEN_HC_HIDDEN   (PULSAR_QWEN_HC * PULSAR_QWEN_HIDDEN)
#define PULSAR_QWEN_HC_LOWRANK  320    /**< hc_lowrank */
#define PULSAR_QWEN_N_EXPERT    512    /**< num_experts */
#define PULSAR_QWEN_TOPK        10     /**< num_experts_per_tok (norm_topk_prob) */
#define PULSAR_QWEN_EXPERT_MID  640    /**< moe_intermediate_size */
#define PULSAR_QWEN_SHARED_MID  640    /**< shared_expert_intermediate_size */
#define PULSAR_QWEN_PLE_HEADS   16     /**< (ngram_size - 1) x heads_per_ngram */
#define PULSAR_QWEN_PLE_ROW     160    /**< ple_embed_dim / PLE_HEADS: one table row */
#define PULSAR_QWEN_PLE_TAPS    4      /**< ple_conv_kernel_size */
#define PULSAR_QWEN_PLE_DIL     3      /**< the conv dilation = ngram_size */
#define PULSAR_QWEN_PLE_STATE   ((PULSAR_QWEN_PLE_TAPS - 1) * PULSAR_QWEN_PLE_DIL)   /**< 9 tokens */

/** L251 MTP: the head-side weights of the input combine (the sidecar's tensors). */
typedef struct {
    const uint16_t *embd;            /**< the trunk's embed_tokens, bf16 [n_vocab][2560] */
    /// Rows of `embd` (the trunk's vocabulary).  The kernel indexes it unclamped: the caller refuses an id outside
    /// [0, n_vocab) on the host before the launch (pulsar_qwen_s4_mtp_combine).
    int n_vocab;
    const uint16_t *norm_embd;       /**< mtp.pre_fc_norm_embedding bf16 [2560] */
    const uint16_t *norm_hidden;     /**< mtp.pre_fc_norm_hidden bf16 [10240] */
    pulsar_mxfp8_lowrank fc_embd;     /**< mtp.fc_embedding mxfp8_lt [2560][2560] */
    pulsar_mxfp8_lowrank fc_hidden;   /**< mtp.fc_hidden mxfp8_lt [2560][2560], applied per stream */
} pulsar_qwen_mtp_dev;

/** The MTP input combine (the L251 MTP spec (2026-09-29, private notes), part (b)): for each row, h_streams [T][4][2560]
 *  bf16 (the trunk's pre-mixer stack at position p) and tokens [T] int32 on the device (x_{p+1}) ->
 *  out_streams [T][4][2560] bf16, the MTP layer's starting streams:
 *    s_i = bf16( bf16(fc_hidden(hn_i)) + bf16(fc_embedding(ein)) ),
 *    hn = (1+w) RMSNorm over all 10240 of h, ein = (1+w) RMSNorm of embed(token).
 *  The workspace is a fixed size (rows go through in slabs). */
/// @return the combine's fixed workspace bytes
size_t pulsar_qwen_mtp_combine_workspace_bytes(void);
/// @param w           the MTP head-side weights
/// @param h_streams   device bf16 [T][4][2560] the trunk's pre-mixer streams
/// @param tokens      device int32 [T] the next tokens, each in [0, w->n_vocab) -- checked by the caller on the host
///                    (the kernel indexes the table unclamped)
/// @param T           rows
/// @param out_streams device bf16 [T][4][2560] the MTP layer's starting streams
/// @param ws          device workspace
/// @param ws_bytes    bytes at ws (>= pulsar_qwen_mtp_combine_workspace_bytes())
/// @param stream      stream to launch on
/// @return 0 on success, negative on a refusal (said) or a launch failure
int pulsar_qwen_mtp_combine_launch(const pulsar_qwen_mtp_dev *w, const uint16_t *h_streams, const int32_t *tokens,
                                   int T, uint16_t *out_streams, void *ws, size_t ws_bytes, cudaStream_t stream);

/** tokens (device i32 [T]) -> streams bf16 [T][4][2560]: each token's
 *  embed_tokens row (bf16 [n_vocab][2560]) repeated into the 4 streams. */
/// @param table   device bf16 [n_vocab][2560] embed_tokens
/// @param tokens  device int32 [T] token ids, each in [0, n_vocab) -- checked by the caller on the host
///                (pulsar_qwen_s4_embed); the kernel indexes the table unclamped
/// @param T       rows
/// @param streams device bf16 [T][4][2560] out
/// @param stream  stream to launch on
/// @return 0, -1 refused (said), -3 launch failure
int pulsar_qwen_embed_launch(const uint16_t *table, const int32_t *tokens, int T, uint16_t *streams,
                             cudaStream_t stream);

/* ======================================================================== */
/* Gated Residual (Qwen4ExpTextGatedResidual) + the top-level mixer           */

/** L251: one Gated Residual site's weights (a block's read + write, or the top-level mixer when inject is NULL). */
typedef struct {
    const uint16_t *norm_w;   /**< hc_norm.weight bf16 [10240]; applied as (1 + w) */
    pulsar_mxfp8_lowrank down; /**< input_mix_weight_down [320][10240] */
    pulsar_mxfp8_lowrank up;   /**< input_mix_weight_up [10240][320]; the same format as down */
    const uint16_t *inject;   /**< block_inject_weight bf16 [4][10240]; NULL = the mixer (no write) */
    /** L266: a PROMPT chunk (a prefill step, not a verify): the prefill GEMM at every row count, so a
     *  prompt cut anywhere is byte-identical to one prefilled whole (session_contract_gate C1).
     *  false: decode widths (<= 16 rows) take the GEMV, wider ones the GEMM. */
    bool prompt;
} pulsar_qwen_gr_dev;

/// Workspace bytes for T rows of a Gated Residual read.
/// @param T rows
/// @return workspace bytes
size_t pulsar_qwen_gr_workspace_bytes(int T);

/** The read: xn = grouped RMSNorm(streams) (1 + w); x = mean_s(sigmoid(W_up
 *  silu(W_down xn / 4)) (.) xn) -> x_bf16 [T][2560] and, when `x` is given,
 *  its E4M3 slot (the mixer before the bf16 head has no A8 consumer and
 *  passes NULL); when the site has a write, inj [T][4] = 2 sigmoid(W_inj xn / 4).
 *  streams bf16 [T][4][2560].  Decode rows (w->prompt false) past PULSAR_ROWS_DECODE_MAX run in blocks of
 *  that many, each the decode arm's bytes (L288 C1). */
/* L251 / ac69748f: x_bf16 only.  The read emits bf16 inside and out -- there is no E4M3 activation
 * slot in this family for it to fill. */
/// @param w        the site's weights
/// @param streams  device bf16 [T][4][2560]
/// @param T        rows
/// @param x_bf16   device bf16 [T][2560] out
/// @param inj      device f32 [T][4] injection weights out (sites with a write), else unused
/// @param ws       device workspace
/// @param ws_bytes bytes at ws (>= pulsar_qwen_gr_workspace_bytes)
/// @param stream   stream to launch on
/// @return 0 on success, negative on a refusal (said) or a launch failure
int pulsar_qwen_gr_read_launch(const pulsar_qwen_gr_dev *w, const uint16_t *streams, int T,
                               uint16_t *x_bf16, float *inj,
                               void *ws, size_t ws_bytes, cudaStream_t stream);

/** The write: streams[t][s] += out[t] * inj[t][s] (f32 math, one bf16 rounding). */
/// @param streams device bf16 [T][4][2560], updated in place
/// @param out     device f32 [T][2560] the block's output
/// @param inj     device f32 [T][4] the read's injection weights
/// @param T       rows
/// @param stream  stream to launch on
/// @return 0 on success, negative on a refusal (said) or a launch failure
int pulsar_qwen_gr_write_launch(uint16_t *streams, const float *out, const float *inj, int T, cudaStream_t stream);

/* ======================================================================== */
/* PLE at layer index 1 (Qwen4ExpTextPLELayer)                               */

/** The layer's weights the launcher reads.  key_proj (2560 -> 10240) and value_proj (2560 -> 2560) are not
 *  here: the caller runs them (L284 #2: through its row-kind segments) and hands in their f32 rows. */
typedef struct {
    const uint16_t *norm_key;        /**< bf16 [10240], (1 + w) */
    const uint16_t *norm_query;      /**< bf16 [10240] */
    const uint16_t *norm_conv;       /**< bf16 [10240] */
    const uint16_t *conv_w;          /**< conv1d.weight bf16 [10240][1][4] */
} pulsar_qwen_ple_dev;

/** Where each batch row sits: rows of one sequence are consecutive and in time
 *  order.  row_seq[r] = the row's sequence q in [0, n_seq), row_j[r] = its
 *  index among that sequence's rows in this batch; seq_first[q] / seq_rows[q]
 *  = the rows of sequence q, seq_bank[q] = the conv-state slot it owns (the
 *  session's bank).  All device arrays. */
typedef struct {
    /** @var row_seq
     *  [T] each row's sequence q in [0, n_seq).
     *  @var row_j
     *  [T] its index among that sequence's rows in this batch.
     *  @var seq_first
     *  [n_seq] sequence q's first batch row.
     *  @var seq_rows
     *  [n_seq] sequence q's rows in this batch.
     *  @var seq_bank
     *  [n_seq] the conv-state slot sequence q owns (the session's bank). */
    const int32_t *row_seq, *row_j, *seq_first, *seq_rows, *seq_bank;
    /// Sequences in the batch.
    int n_seq;
    /** L251 MTP verify (NULL otherwise): the conv state AFTER each row of a sequence but its last, at the
     *  row's batch index (L272 P1 S4: any number of sequences), [rows][PULSAR_QWEN_PLE_STATE][10240] f32 --
     *  each sequence's last-row slot unused -- so a rejected draft rolls back by copying a row's back.
     *  The arithmetic is unchanged. */
    float *state_rows;
    /** L284 #2: the batch rows [0, n_state_rows) state_rows captures -- a fused step's verify rows; the prompt
     *  run behind them is not a verify's (state_rows holds SPEC_ROWS rows). */
    int n_state_rows;
} pulsar_qwen_rows;

/** Workspace bytes for T rows: a function of the shape alone. */
/// @param T rows
/// @return workspace bytes
size_t pulsar_qwen_ple_workspace_bytes(int T);

/** The PLE injection for T rows, from key f32 [T][10240] = key_proj e and value f32 [T][2560] = value_proj e,
 *  e the 16 gathered bf16 table rows per token (head order); streams bf16 [T][4][2560] updated in place:
 *    k = norm_key(key), q_s = norm_query(stream_s), v = value,
 *    gate_s = signed-sqrt(<k_s, q_s> / sqrt(2560)), gv_s = sigmoid(gate_s) v,
 *    stream_s += gv_s + silu(conv_dil3(norm_conv(gv)))
 *  conv_state f32 [banks][9][10240] (oldest first; slot seq_bank[q] for
 *  sequence q) is read for taps before the batch and advanced by it. */
/// @param w          the layer's norm and conv weights
/// @param key        device f32 [T][10240] key_proj rows
/// @param value      device f32 [T][2560] value_proj rows
/// @param streams    device bf16 [T][4][2560], updated in place
/// @param T          rows
/// @param rows       where each row sits (sequences, banks, verify captures)
/// @param conv_state device f32 [banks][9][10240] conv state, advanced
/// @param ws         device workspace
/// @param ws_bytes   bytes at ws (>= pulsar_qwen_ple_workspace_bytes)
/// @param stream     stream to launch on
/// @return 0 on success, negative on a refusal (said) or a launch failure
int pulsar_qwen_ple_launch(const pulsar_qwen_ple_dev *w, const float *key, const float *value, uint16_t *streams,
                           int T, const pulsar_qwen_rows *rows, float *conv_state,
                           void *ws, size_t ws_bytes, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
#endif /* PULSAR_CUDA_QWEN_H */

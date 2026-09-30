/* Qwen3.8-Flash-Next (HF `qwen4_exp`) block pieces -- L251 stream S4.
 *
 * The MoE block (softmax top-10 router, EXL3 routed experts, sigmoid-gated
 * shared expert), the Gated Residual (GR: the 4-stream low-rank read, the
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

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
/* The engine's family ops (src/engine/family_qwen_s4.cpp) call these
 * launchers from a CUDA-free TU: the runtime's own typedef is all they need
 * (they pass 0, the per-thread default stream the engine runs on). */
typedef struct CUstream_st *cudaStream_t;
#endif
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct pulsar_tessera_proj;   /* src/cuda/mmq/pulsar_tessera.h (L255) */

/* ---- the shapes these kernels are built for (config.json of the checkpoint).
 * The launchers take the dims they are handed and REFUSE any that differ: the
 * config is the authority, these constants are what the templates were
 * instantiated for. */
#define PULSAR_QWEN_HIDDEN      2560   /**< hidden_size */
#define PULSAR_QWEN_HC          4      /**< hc_count: residual streams */
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

/** An E4M3 activation slot: rows x dim E4M3 row-major plus one ue8m0 byte per
 *  32 at pulsar_mx_sfoff(row, k / 32, kbp), kbp = pulsar_mx_kbp(dim) -- the
 *  layout every A8 GEMV in the engine reads.  The scale slab spans
 *  pulsar_mx_sf_slab_bytes(rows, kbp) bytes and must be zeroed before a
 *  producer writes it (the swizzle leaves holes). */
typedef struct {
    uint8_t *q;
    uint8_t *sf;
    int      kbp;
} pulsar_qwen_slot;

/** A gated-residual low-rank weight, [out][in] row-major, in one of the two
 *  formats the recipe stores them in:
 *    MXFP8_LT (sf != NULL): E4M3 codes then one E8M0 byte per (row, 32 inputs)
 *      at pulsar_mx_sfoff(row, k / 32, pulsar_mx_kbp(in)) -- every per-layer site;
 *    BF16     (sf == NULL): the top-level mixer (the graded recipe keeps it bf16).
 *  The activation each one reads follows its format (rule 3): E4M3 per 32 for
 *  MXFP8, bf16 for BF16 -- the read emits the one its weights take. */
typedef struct {
    const void *w;
    const uint8_t *sf;
    int out, in;
} pulsar_qwen_lowrank;

/** A dense Linear in the EXL3 format: one [trellis | suh | svh] slice in
 *  exl3_expert_layout's byte model (so an exllamav3 checkpoint's tensors copy
 *  in verbatim), run by the EXL3 dense arm (mmq/ds4_exl3_dense.cuh). */
typedef struct {
    const void *w;
    const uint8_t *sf;   /**< the mxfp8_lt E8M0 plane (NULL for EXL3): the
                          *  recipe's dense tier is MIXED, so the arm follows
                          *  k2 -- 0 means mxfp8_lt, 4..10 an EXL3 rate. */
    int k2;          /**< rate in half-bit units, 4..10; 0 = mxfp8_lt */
    int in, out;
} pulsar_qwen_linear;

/** L251 MTP: the head-side weights of the input combine (the sidecar's tensors). */
typedef struct {
    const uint16_t *embd;            /**< the trunk's embed_tokens, bf16 [n_vocab][2560] */
    int n_vocab;
    const uint16_t *norm_embd;       /**< mtp.pre_fc_norm_embedding bf16 [2560] */
    const uint16_t *norm_hidden;     /**< mtp.pre_fc_norm_hidden bf16 [10240] */
    pulsar_qwen_lowrank fc_embd;     /**< mtp.fc_embedding mxfp8_lt [2560][2560] */
    pulsar_qwen_lowrank fc_hidden;   /**< mtp.fc_hidden mxfp8_lt [2560][2560], applied per stream */
} pulsar_qwen_mtp_dev;

/** The MTP input combine (l251/docs/MTP-SPEC-2026-09-29.md (b)): for each row, h_streams [T][4][2560]
 *  bf16 (the trunk's pre-mixer stack at position p) and tokens [T] int32 on the device (x_{p+1}) ->
 *  out_streams [T][4][2560] bf16, the MTP layer's starting streams:
 *    s_i = bf16( bf16(fc_hidden(hn_i)) + bf16(fc_embedding(ein)) ),
 *    hn = (1+w) RMSNorm over all 10240 of h, ein = (1+w) RMSNorm of embed(token).
 *  The workspace is a fixed size (rows go through in slabs). */
size_t pulsar_qwen_mtp_combine_workspace_bytes(void);
int pulsar_qwen_mtp_combine_launch(const pulsar_qwen_mtp_dev *w, const uint16_t *h_streams, const int32_t *tokens,
                                   int T, uint16_t *out_streams, void *ws, size_t ws_bytes, cudaStream_t stream);

/** A plain MXFP8 dense Linear: y [rows][out] f32 = W x, W stored mxfp8_lt
 *  ([out][in] E4M3 then the swizzled E8M0 plane) and x the block input's bf16 row
 *  of width `l->in` -- there is no E4M3 activation slot in this family (L251 /
 *  ac69748f).  The same arithmetic the GR arm's W_down uses (split-K, one warp
 *  per output row, ordered); no workspace. */
int pulsar_qwen_mxfp8_linear_launch(const pulsar_qwen_lowrank *l, const uint16_t *x_bf16, int rows, float *y,
                                    void *ws, size_t ws_bytes, cudaStream_t stream);
/** L251: bytes of an [out][in] mxfp8_lt matrix (E4M3 then the swizzled E8M0 plane); 0 if in % 32. */
uint64_t pulsar_qwen_mxfp8_bytes(int out, int in);
/** L251: a bf16 [out][in] matrix into mxfp8_lt at `dst` (pulsar_qwen_mxfp8_bytes), with the
 *  producers' own encoder.  `rows` (device, [out], or NULL for the identity) gathers: output row r is
 *  w's row rows[r] (the MTP draft head, L251).  0 on success. */
int pulsar_qwen_bf16_to_mxfp8(const uint16_t *w, const int32_t *rows, int out, int in, void *dst, cudaStream_t stream);

/** A weight's device pointer: the engine's model-range cache for the span
 *  [offset, offset + bytes) of `model_map` (cuda_model_range_ptr). */
const void *pulsar_qwen_weight_ptr(const void *model_map, uint64_t offset, uint64_t bytes, const char *what);

/** The device table of [trellis, scales] pointer pairs over an EXL3 expert
 *  stack (exl3_expert_table): n_expert slices of `stride` bytes, the scales
 *  plane at `split` into each. */
const void *const *pulsar_qwen_expert_table(const void *stack, uint32_t n_expert, uint64_t stride, uint64_t split);

/** tokens (device i32 [T]) -> streams bf16 [T][4][2560]: each token's
 *  embed_tokens row (bf16 [n_vocab][2560]) repeated into the 4 streams. */
int pulsar_qwen_embed_launch(const uint16_t *table, const int32_t *tokens, int T, int n_vocab, uint16_t *streams,
                             cudaStream_t stream);

/** Workspace bytes pulsar_qwen_linear_launch needs for `rows` rows. */
size_t pulsar_qwen_linear_workspace_bytes(const pulsar_qwen_linear *l, int rows);
/** y [rows][out] f32 = the complete Linear of the bf16 rows in `x_bf16`. */
int pulsar_qwen_linear_launch(const pulsar_qwen_linear *l, const uint16_t *x_bf16, int rows, float *y,
                              void *ws, size_t ws_bytes, cudaStream_t stream);

/* ======================================================================== */
/* MoE block (Qwen4ExpTextSparseMoeBlock)                                     */

/** The router alone: logits = W_r x (bf16 x bf16, f32 accumulate in a fixed
 *  order), rounded to bf16; softmax over the experts; the top-k by probability
 *  (ties to the LOWER expert id); weights renormalised over the k and rounded
 *  to bf16.  Row PULSAR_QWEN_N_EXPERT of `logits` is the shared-expert gate
 *  logit (w_sg . x), and `sgate` = sigmoid of its bf16 rounding.
 *  x_bf16 [T][2560]; router_w bf16 [512][2560]; shared_gate_w bf16 [2560];
 *  logits [T][513] f32 (workspace, left readable); selected [T][10] int32 in
 *  descending probability; weights [T][10] f32 (bf16 values); sgate [T]. */
int pulsar_qwen_router_launch(const uint16_t *x_bf16, const uint16_t *router_w, const uint16_t *shared_gate_w,
                              int T, int hidden, int n_expert, int top_k,
                              float *logits, int32_t *selected, float *weights, float *sgate,
                              cudaStream_t stream);

typedef struct {
    const uint16_t *router_w;        /**< bf16 [512][2560] */
    const uint16_t *shared_gate_w;   /**< bf16 [2560] */
    const void *const *gate_up_table;/**< exl3_expert_table pairs [512][2]: the FUSED gate_up 2560 -> 1280
                                          (output rows 0..639 gate, 640..1279 up -- the container's layout) */
    const void *const *down_table;   /**< down_proj 640 -> 2560 */
    /** L251 MTP: the SPLIT form -- gate 2560 -> 640 and up 2560 -> 640 as two slices, each with its own
     *  suh (the MTP layer's experts, turboderp's EXL3).  Exactly one of gate_up_table and
     *  (gate_table, up_table) is set; the artifact decides, the launcher refuses anything else. */
    const void *const *gate_table, *const *up_table;
    /** L255: the routed experts in Tessera's value family (src/cuda/mmq/pulsar_tessera.h) -- gate, up and down
     *  stacks of n_expert, kernel-ready.  Set instead of every EXL3 table above when the artifact carries the
     *  layer's experts that way; the launcher refuses a block with both or neither. */
    const struct pulsar_tessera_proj *tessera_gate, *tessera_up, *tessera_down;
    int k2_gate_up, k2_down;         /**< routed rates (half-bit units); k2_gate_up is the pair's rate in the split form */
    pulsar_qwen_linear shared_gate, shared_up, shared_down;
} pulsar_qwen_moe_dev;

/** Workspace bytes for T rows (everything but the MMQ drivers' own arena):
 *  a function of the shape alone. */
size_t pulsar_qwen_moe_workspace_bytes(int T);

/** The MoE block: out [T][2560] f32 = sum over the top-10 (slot order) of
 *  w_k * down_k(silu(gate_k x) * up_k x) + sigmoid(w_sg . x) * shared(x).
 *  `x_bf16` is the block input row and `x` its E4M3 slot (both from the GR
 *  read).  Non-finite outputs record `nf_code` in *nf_flag (first writer
 *  wins).  Needs the MMQ drivers (PULSAR_HAVE_MMQ); refuses without them. */
/* L251 / ac69748f: x_bf16 only.  There is no E4M3 activation slot in this family, so the MoE takes
 * the block input's bf16 row and nothing else -- the routed arm reads it by ids_src1 and the shared
 * expert reads it directly. */
int pulsar_qwen_moe_launch(const pulsar_qwen_moe_dev *w, const uint16_t *x_bf16,
                           int T, float *out, void *ws, size_t ws_bytes,
                           uint32_t *nf_flag, uint32_t nf_code, cudaStream_t stream);

/* ======================================================================== */
/* Gated Residual (Qwen4ExpTextGatedResidual) + the top-level mixer           */

typedef struct {
    const uint16_t *norm_w;   /**< hc_norm.weight bf16 [10240]; applied as (1 + w) */
    pulsar_qwen_lowrank down; /**< input_mix_weight_down [320][10240] */
    pulsar_qwen_lowrank up;   /**< input_mix_weight_up [10240][320]; the same format as down */
    const uint16_t *inject;   /**< block_inject_weight bf16 [4][10240]; NULL = the mixer (no write) */
} pulsar_qwen_gr_dev;

size_t pulsar_qwen_gr_workspace_bytes(int T);

/** The read: xn = grouped RMSNorm(streams) (1 + w); x = mean_s(sigmoid(W_up
 *  silu(W_down xn / 4)) (.) xn) -> x_bf16 [T][2560] and, when `x` is given,
 *  its E4M3 slot (the mixer before the bf16 head has no A8 consumer and
 *  passes NULL); when the site has a write, inj [T][4] = 2 sigmoid(W_inj xn / 4).
 *  streams bf16 [T][4][2560]. */
/* L251 / ac69748f: x_bf16 only.  The read emits bf16 inside and out -- there is no E4M3 activation
 * slot in this family for it to fill. */
int pulsar_qwen_gr_read_launch(const pulsar_qwen_gr_dev *w, const uint16_t *streams, int T,
                               uint16_t *x_bf16, float *inj,
                               void *ws, size_t ws_bytes, cudaStream_t stream);

/** The write: streams[t][s] += out[t] * inj[t][s] (f32 math, one bf16 rounding). */
int pulsar_qwen_gr_write_launch(uint16_t *streams, const float *out, const float *inj, int T, cudaStream_t stream);

/* ======================================================================== */
/* PLE at layer index 1 (Qwen4ExpTextPLELayer)                               */

typedef struct {
    pulsar_qwen_linear key_proj;     /**< 2560 -> 10240 */
    pulsar_qwen_linear value_proj;   /**< 2560 -> 2560 */
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
    const int32_t *row_seq, *row_j, *seq_first, *seq_rows, *seq_bank;
    int n_seq;
    /** L251 MTP verify (NULL otherwise; set only with n_seq 1): the conv state AFTER each row
     *  r < rows - 1, [rows - 1][PULSAR_QWEN_PLE_STATE][10240] f32, so a rejected draft rolls back by
     *  copying row r's back.  The arithmetic is unchanged. */
    float *state_rows;
} pulsar_qwen_rows;

/** Workspace bytes for T rows: a function of the shape alone. */
size_t pulsar_qwen_ple_workspace_bytes(int T);

/** The PLE injection for T rows: `emb` bf16 [T][2560] = the 16 gathered table
 *  rows per token (head order); streams bf16 [T][4][2560] updated in place:
 *    k = norm_key(key_proj e), q_s = norm_query(stream_s), v = value_proj e,
 *    gate_s = signed-sqrt(<k_s, q_s> / sqrt(2560)), gv_s = sigmoid(gate_s) v,
 *    stream_s += gv_s + silu(conv_dil3(norm_conv(gv)))
 *  conv_state f32 [banks][9][10240] (oldest first; slot seq_bank[q] for
 *  sequence q) is read for taps before the batch and advanced by it. */
int pulsar_qwen_ple_launch(const pulsar_qwen_ple_dev *w, const uint16_t *emb, uint16_t *streams, int T,
                           const pulsar_qwen_rows *rows, float *conv_state,
                           void *ws, size_t ws_bytes, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* PULSAR_CUDA_QWEN_H */

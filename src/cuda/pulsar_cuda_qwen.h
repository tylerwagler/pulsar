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

#include "pulsar_gpu.h"   /* PULSAR_GPU_MNEUTRAL_ROWS_MAX */
#include "pulsar_cuda_router.h"

#ifdef __cplusplus
extern "C" {
#endif

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
    /** L266: a PROMPT chunk (a prefill step, not a verify): the prefill GEMM at every row count, so a
     *  prompt cut anywhere is byte-identical to one prefilled whole (session_contract_gate C1).
     *  false: decode widths (<= 16 rows) take the GEMV, wider ones the GEMM. */
    bool prompt;
} pulsar_qwen_lowrank;

/** L272 P4c: the bf16-ROWS dense linear's reference (family-neutral; built by the core's front door,
 *  pulsar_linear_rows_ref in linear.cpp, from the format table): an mxfp8_lt weight (the W8A16 arm) or an
 *  EXL3 one -- one [trellis | suh | svh] slice in exl3_expert_layout's byte model, so an exllamav3
 *  checkpoint's tensors copy in verbatim (mmq/ds4_exl3_dense.cuh). */
typedef struct pulsar_rows_linear {
    const void *w;
    const uint8_t *sf;   /**< the mxfp8_lt E8M0 plane (NULL for EXL3): the
                          *  recipe's dense tier is MIXED, so the arm follows
                          *  k2 -- 0 means mxfp8_lt, 4..10 an EXL3 rate. */
    int k2;          /**< rate in half-bit units, 4..10; 0 = mxfp8_lt */
    int in, out;
    /** L266: a PROMPT chunk (a prefill step, not a verify): the prefill GEMM at every row count, so a
     *  prompt cut anywhere is byte-identical to one prefilled whole (session_contract_gate C1).
     *  false: decode rows -- the decode arms at every width, in blocks of PULSAR_QWEN_DECODE_ROWS_MAX past it
     *  (L288 C1), so a row's bytes never depend on how many ride with it. */
    bool prompt;
    /** L295: a bf16 [out][in] weight (k2 0, sf NULL): the fixed-order bf16 rows product -- the shared router's logits
     *  kernel (pulsar_router_logits_launch), one f32 accumulation order per (row, output) at any row count, so a
     *  row's output is the same bytes for decode, verify and prompt rows alike (`prompt` changes nothing). */
    bool bf16;
} pulsar_rows_linear;

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

/** L289: one draft-head row's greedy pick and its confidence, reduced on the device so a draft step reads back
 *  this record instead of the row (263 KB) and the host no longer scans it.  L288: up to
 *  PULSAR_QWEN_DRAFT_TOP_ROWS_MAX rows a launch (every bank's draft row of a step), one record each -- at least the
 *  head's rows (static_assert in family_qwen_s4.cpp; L288 C1: 32, the shared verify width). */
#define PULSAR_QWEN_DRAFT_TOP_ROWS_MAX 32
typedef struct {
    int32_t index;      /**< the FIRST index of the row's maximum (a strict > scan from index 0) */
    float conf;         /**< (float)(1.0 / sum_i exp((double)(row[i] - row[index]))): its softmax probability */
    uint32_t nan;       /**< nonzero: the row holds a NaN -- the pick means nothing and the caller refuses */
    uint32_t pad;
} pulsar_qwen_draft_top;
/** Device bytes of the reduction's workspace (PULSAR_QWEN_DRAFT_TOP_ROWS_MAX records first, row 0's at offset 0). */
size_t pulsar_qwen_draft_top_ws_bytes(void);
/** rows [n_rows][n] f32 (device) -> record r at the start of `ws` (device, pulsar_qwen_draft_top_ws_bytes) is row r's.
 *  The pick is the host scan's (ties to the lower index); the sum is in double, each term exp((double)(row[i] - m))
 *  with the f32 difference the host rule takes, so the f32 confidence is the host rule's value; a row's record does
 *  not depend on the rows beside it.  n_rows 1 .. PULSAR_QWEN_DRAFT_TOP_ROWS_MAX.  0 on success. */
int pulsar_qwen_draft_top_launch(const float *rows, int n, int n_rows, void *ws, cudaStream_t stream);

/** The widest call the W8A16 decode arms take (the rows kernel's two n8 token tiles, the GEMV's staging) -- the
 *  M-independent width: decode rows past it go through the front door (pulsar_rows_linear_launch) and the GR read in
 *  calls of at most this many, each row the bytes a one-row decode gives (L288 C1). */
#define PULSAR_QWEN_DECODE_ROWS_MAX ((int)PULSAR_GPU_MNEUTRAL_ROWS_MAX)

/** A plain MXFP8 dense Linear: y [rows][out] f32 = W x, W stored mxfp8_lt
 *  ([out][in] E4M3 then the swizzled E8M0 plane) and x the block input's bf16 row
 *  of width `l->in` -- there is no E4M3 activation slot in this family (L251 /
 *  ac69748f).  At decode widths (<= 16 rows) a Linear of >= 256 outputs takes the tensor-core rows kernel the
 *  head uses (L288: the weight read once for all rows), a narrower one the GEMV (one warp per output row,
 *  ordered); prompt widths the tensor-core GEMM.  No workspace. */
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
const void *pulsar_gpu_weight_range_ptr(const void *model_map, uint64_t offset, uint64_t bytes, const char *what);

/** The device table of [trellis, scales] pointer pairs over an EXL3 expert
 *  stack (exl3_expert_table): n_expert slices of `stride` bytes, the scales
 *  plane at `split` into each. */
const void *const *pulsar_exl3_expert_table(const void *stack, uint32_t n_expert, uint64_t stride, uint64_t split);

/** tokens (device i32 [T]) -> streams bf16 [T][4][2560]: each token's
 *  embed_tokens row (bf16 [n_vocab][2560]) repeated into the 4 streams. */
int pulsar_qwen_embed_launch(const uint16_t *table, const int32_t *tokens, int T, int n_vocab, uint16_t *streams,
                             cudaStream_t stream);

/** Workspace bytes pulsar_rows_linear_launch needs for `rows` rows. */
size_t pulsar_rows_linear_workspace_bytes(const pulsar_rows_linear *l, int rows);
/** y [rows][out] f32 = the complete Linear of the bf16 rows in `x_bf16`. */
int pulsar_rows_linear_launch(const pulsar_rows_linear *l, const uint16_t *x_bf16, int rows, float *y,
                              void *ws, size_t ws_bytes, cudaStream_t stream);

/** L288 exl3-dense: workspace bytes pulsar_rows_linear_multi_launch needs for n Linears of one input at `rows`
 *  rows -- each 128-multiple width sized as an EXL3 projection, the most it takes (an mxfp8_lt one runs alone
 *  after the EXL3 launch and reuses it). */
size_t pulsar_rows_linear_multi_workspace_bytes(const pulsar_rows_linear *const *ls, int n, int rows);
/** L288 exl3-dense: y[j] [rows][ls[j]->out] f32 = Linear j of the SAME bf16 rows, for n (1..4) Linears of one
 *  input width and row kind -- the EXL3 ones in one dense multi launch (one prep, one GEMV per rate: each output
 *  bit for bit what pulsar_rows_linear_launch writes for it alone), then each mxfp8_lt one by itself. */
int pulsar_rows_linear_multi_launch(const pulsar_rows_linear *const *ls, int n, const uint16_t *x_bf16, int rows,
                                    float *const *y, void *ws, size_t ws_bytes, cudaStream_t stream);

/* ======================================================================== */
/* MoE block (Qwen4ExpTextSparseMoeBlock)                                     */

/** L295 F2: the family's router, declared for the shared router (pulsar_cuda_router.h): logits = W_r x (bf16 x
 *  bf16, f32 in a fixed order) rounded to bf16; softmax over the experts; the top-k by probability (ties to the
 *  LOWER expert id); weights renormalised over the k (no floor, no scale) and rounded to bf16.  One shared expert,
 *  gated by its own logit row (W_r's row n_expert is shared_expert_gate): gamma = bf16(sigmoid(bf16 logit)). */
static inline pulsar_router_cfg pulsar_qwen_router_cfg(int n_expert, int top_k) {
    pulsar_router_cfg c;
    c.n_expert = n_expert;
    c.top_k = top_k;
    c.n_shared = 1;
    c.shared_gate = PULSAR_ROUTER_SHARED_SIGMOID;
    c.score = PULSAR_ROUTER_SCORE_SOFTMAX;
    c.renorm = PULSAR_ROUTER_RENORM_SUM;
    c.logits_bf16 = true;
    c.weights_bf16 = true;
    c.sum_floor = 0.0f;
    c.route_scale = 1.0f;
    return c;
}

/** L272 P4c: the routed MoE over bf16 ROWS -- the core front door's arm (moe.cpp builds it from the expert
 *  stacks' formats; family-neutral).  EXL3 experts: a FUSED gate_up stack (output rows 0..mid-1 gate, mid..2mid-1
 *  up) or a gate + up PAIR (two stacks, each with its own suh), and a down stack.  The tables hold experts
 *  [ex_lo, ex_lo + n_local) -- all of them on one GPU; under expert parallelism a token's picks of another
 *  rank's experts are dropped here and summed there. */
struct pulsar_tessera_proj;   /* src/cuda/mmq/pulsar_tessera.h (L255) */
/** One shared expert: out += scale[t] * down(silu(gate x) * up x), added after the routed sum (a block's N shared
 *  experts in order).  Task #44: one whose three Linears are EXL3 dense slices rides the routed launches (the EXL3
 *  routed front door, mmq/exl3_moe.cuh) -- keyed on that format; any other format runs
 *  pulsar_rows_moe_shared_launch. */
typedef struct pulsar_rows_moe_shared {
    pulsar_rows_linear gate, up, down;   /**< EXL3 (k2 != 0) all three to ride */
    const float *scale;                  /**< [T] the shared term's multiplier (its router gamma), or NULL = 1 */
    float *yg, *yu, *ys;                 /**< [T][mid], [T][mid], [T][out] f32 */
    uint16_t *h;                         /**< [T][mid] bf16: the SwiGLU rows the down reads (the imatrix tap's) */
    void *ws;                            /**< the dense arm's workspace, pulsar_rows_moe_shared_ws_bytes */
    size_t ws_bytes;
} pulsar_rows_moe_shared;
/** The dense-arm workspace a riding shared expert needs for T rows (in -> mid -> out). */
size_t pulsar_rows_moe_shared_ws_bytes(int T, int in, int mid, int out);
/** L288 exl3-dense: fork the riding shared expert of T bf16 rows onto its own stream (exl3_moe_shared_fork) --
 *  call it AHEAD of the router, then pulsar_rows_moe_routed_launch with the same descriptor joins it before its
 *  sum.  Each pending fork needs its own yg / yu / ys / h rows; a block's shared experts fork in their order.
 *  0 on success. */
int pulsar_rows_moe_shared_fork(const pulsar_rows_moe_shared *q, const uint16_t *x_bf16, int T, cudaStream_t stream);
/** A shared expert that does not ride (another format, or a Tessera routed arm), after the routed sum is in `out`:
 *  out += scale * down(silu(gate x) * up x), the three linears through the front door's references
 *  (pulsar_linear_rows_ref), over q's buffers. */
int pulsar_rows_moe_shared_launch(const pulsar_rows_moe_shared *q, const uint16_t *x_bf16, int T, float *out,
                                  cudaStream_t stream);
typedef struct pulsar_rows_moe {
    const void *const *gate_up_table;   /**< exl3_expert_table pairs, the fused form, or NULL */
    const void *const *gate_table, *const *up_table;   /**< the pair form, or NULL */
    const void *const *down_table;
    /** L255: the routed experts in Tessera's value family (src/cuda/mmq/pulsar_tessera.h) -- gate, up and down
     *  stacks of n_expert, kernel-ready.  Set INSTEAD of every EXL3 table above; the launcher refuses a descriptor
     *  with both or neither, and (no expert range in that arm yet) one whose tables are not every expert. */
    const struct pulsar_tessera_proj *tessera_gate, *tessera_up, *tessera_down;
    int k2_gate_up, k2_down;             /**< rates (half-bit units); k2_gate_up is the pair's in the split form */
    int n_expert, top_k;                 /**< the model's experts (the router's picks index them) and picks per row */
    int in_dim, mid_dim, out_dim;        /**< the expert shape (the model's): in == out, the block's hidden */
    int ex_lo, n_local;                  /**< the experts the tables hold */
    const pulsar_rows_moe_shared *shared; /**< [n_shared] EXL3 shared experts riding the launches (each forked
                                           *  first), added after the routed sum in order; NULL when none */
    int n_shared;                        /**< 0..PULSAR_ROUTER_SHARED_MAX */
    /** a PROMPT chunk (not a verify): the routed prefill GEMM at every row count, so a prompt cut anywhere is
     *  byte-identical to one prefilled whole (session_contract_gate C1); false: decode rows, the decode GEMM. */
    bool prompt;
} pulsar_rows_moe;
/** Workspace bytes for T rows of the routed part. */
size_t pulsar_rows_moe_routed_workspace_bytes(int T, int top_k, int mid, int out_dim);
/** The bf16 rows [T * k][mid] the last pulsar_rows_moe_routed_launch left in `ws` -- each pick's SwiGLU row with
 *  its route weight folded in, the down projection's input (the importance-matrix collection reads it). */
const uint16_t *pulsar_rows_moe_routed_mid(void *ws, size_t ws_bytes, int T, int top_k, int mid, int out_dim);
/** out [T][hidden] f32 = sum over each row's top-k picks (`sel` [T][k], localised IN PLACE under expert
 *  parallelism; `wts` [T][k]) of w_k * down_k(silu(gate_k x) * up_k x), x the bf16 rows.  Non-finite outputs
 *  record `nf_code` in *nf_flag (first writer wins).  Needs the MMQ drivers (PULSAR_HAVE_MMQ). */
int pulsar_rows_moe_routed_launch(const pulsar_rows_moe *w, int32_t *sel, const float *wts, const uint16_t *x_bf16,
                                  int T, float *out, void *ws, size_t ws_bytes, uint32_t *nf_flag, uint32_t nf_code,
                                  cudaStream_t stream);

/** L295 F2: a rows MoE block's shape (the model's): the router's logit rows (pulsar_router_logit_rows), the routed
 *  experts, and N shared experts. */
typedef struct pulsar_moe_block_shape {
    int n_expert, top_k, n_logit_rows;
    int hidden, expert_mid;
    int n_shared, shared_mid;     /**< 0..PULSAR_ROUTER_SHARED_MAX shared experts, hidden -> shared_mid -> hidden */
} pulsar_moe_block_shape;
/** The block's workspace, carved: the router's outputs, the routed part's region, the shared experts'. */
typedef struct pulsar_moe_block_parts {
    float *logits;      /**< [T][n_logit_rows] the router's logits */
    int32_t *sel;       /**< [T][top_k] */
    float *wts;         /**< [T][top_k] */
    float *gammas;      /**< [n_shared][T] the shared experts' gates (pulsar_router_select_launch's planes) */
    void *routed;       /**< pulsar_rows_moe_routed_launch's workspace */
    size_t routed_bytes;
    float *yg, *yu;     /**< [n_shared][T][shared_mid] */
    float *ys;          /**< [n_shared][T][hidden] */
    uint16_t *h;        /**< [n_shared][T][shared_mid] the shared SwiGLUs' bf16 rows */
    void *lin;          /**< the shared experts' dense-arm workspace (one, shared) */
    size_t lin_bytes;
} pulsar_moe_block_parts;
/** Workspace bytes for T rows of the whole block (a function of the shape alone); 0 for no shape. */
size_t pulsar_moe_block_workspace_bytes(const pulsar_moe_block_shape *s, int T);
int pulsar_moe_block_carve(const pulsar_moe_block_shape *s, int T, void *ws, size_t ws_bytes,
                           pulsar_moe_block_parts *parts);

/* ======================================================================== */
/* Gated Residual (Qwen4ExpTextGatedResidual) + the top-level mixer           */

typedef struct {
    const uint16_t *norm_w;   /**< hc_norm.weight bf16 [10240]; applied as (1 + w) */
    pulsar_qwen_lowrank down; /**< input_mix_weight_down [320][10240] */
    pulsar_qwen_lowrank up;   /**< input_mix_weight_up [10240][320]; the same format as down */
    const uint16_t *inject;   /**< block_inject_weight bf16 [4][10240]; NULL = the mixer (no write) */
    /** L266: a PROMPT chunk (a prefill step, not a verify): the prefill GEMM at every row count, so a
     *  prompt cut anywhere is byte-identical to one prefilled whole (session_contract_gate C1).
     *  false: decode widths (<= 16 rows) take the GEMV, wider ones the GEMM. */
    bool prompt;
} pulsar_qwen_gr_dev;

size_t pulsar_qwen_gr_workspace_bytes(int T);

/** The read: xn = grouped RMSNorm(streams) (1 + w); x = mean_s(sigmoid(W_up
 *  silu(W_down xn / 4)) (.) xn) -> x_bf16 [T][2560] and, when `x` is given,
 *  its E4M3 slot (the mixer before the bf16 head has no A8 consumer and
 *  passes NULL); when the site has a write, inj [T][4] = 2 sigmoid(W_inj xn / 4).
 *  streams bf16 [T][4][2560].  Decode rows (w->prompt false) past PULSAR_QWEN_DECODE_ROWS_MAX run in blocks of
 *  that many, each the decode arm's bytes (L288 C1). */
/* L251 / ac69748f: x_bf16 only.  The read emits bf16 inside and out -- there is no E4M3 activation
 * slot in this family for it to fill. */
int pulsar_qwen_gr_read_launch(const pulsar_qwen_gr_dev *w, const uint16_t *streams, int T,
                               uint16_t *x_bf16, float *inj,
                               void *ws, size_t ws_bytes, cudaStream_t stream);

/** The write: streams[t][s] += out[t] * inj[t][s] (f32 math, one bf16 rounding). */
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
    const int32_t *row_seq, *row_j, *seq_first, *seq_rows, *seq_bank;
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
size_t pulsar_qwen_ple_workspace_bytes(int T);

/** The PLE injection for T rows, from key f32 [T][10240] = key_proj e and value f32 [T][2560] = value_proj e,
 *  e the 16 gathered bf16 table rows per token (head order); streams bf16 [T][4][2560] updated in place:
 *    k = norm_key(key), q_s = norm_query(stream_s), v = value,
 *    gate_s = signed-sqrt(<k_s, q_s> / sqrt(2560)), gv_s = sigmoid(gate_s) v,
 *    stream_s += gv_s + silu(conv_dil3(norm_conv(gv)))
 *  conv_state f32 [banks][9][10240] (oldest first; slot seq_bank[q] for
 *  sequence q) is read for taps before the batch and advanced by it. */
int pulsar_qwen_ple_launch(const pulsar_qwen_ple_dev *w, const float *key, const float *value, uint16_t *streams,
                           int T, const pulsar_qwen_rows *rows, float *conv_state,
                           void *ws, size_t ws_bytes, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* PULSAR_CUDA_QWEN_H */

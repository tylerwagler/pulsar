/* Gated DeltaNet (L251, Qwen3.8-Flash-Next): everything BETWEEN the block's
 * projections for the 36 linear-attention layers -- the causal conv (and its
 * 3-token state), SiLU, the q/k L2 norms, the decay/beta gates, the delta-rule
 * recurrence (fp32 state, 48 V heads x 128 x 128) and the sigmoid-gated
 * RMSNorm.  The Linears (in_proj_qkv / _z / _a / _b, out_proj) are not here.
 *
 * The reference is transformers' qwen4_exp `Qwen4ExpTextGatedDeltaNet`:
 *
 *   x = SiLU(conv1d_k4(qkv))                       depthwise, no bias, causal
 *   q, k, v = split(x, 2048, 2048, 6144)           16 K heads, 48 V heads, 128 each
 *   q = l2norm(q) / sqrt(128), k = l2norm(k)       eps 1e-6 inside the rsqrt
 *   V head h reads K head h / 3                    (repeat_interleave)
 *   beta  = sigmoid(b)
 *   decay = exp(-exp(A_log) * softplus(a + dt_bias))
 *   S = decay * S;  delta = beta * (v - S^T k);  S += k delta^T;  o = S^T q
 *   y = norm_w * o * rsqrt(mean(o^2) + 1e-6) * sigmoid(z)     per V head (128)
 *
 * ONE ARITHMETIC FOR EVERY SCHEDULE.  Prefill is the same per-token step as
 * decode, run by the same kernel with the same thread mapping, carrying the
 * state in registers across a sequence's tokens and through global memory
 * between calls.  So decode of N tokens, a prefill of N tokens, and a prefill
 * split at ANY points are bit-identical -- outputs and both states -- by
 * construction, and a sequence's bytes never depend on its batchmates (one CTA
 * per (sequence, head)).  tests/gdn_gate.cu asserts all three.  The chunkwise
 * (WY / UT-transform) algorithm HF uses for prefill is a different
 * association of the same sums; it is not built (see the L251 GDN notes for
 * when it would pay).
 *
 * Two kernels per call:
 *   1. conv + SiLU + L2 norms + gates  -> the call's scratch (f32 q^, k^, v,
 *      decay, beta per row).  One CTA per (sequence, 16-token tile, 128-channel
 *      head); the tile-0 CTA also advances the sequence's conv state, reading
 *      the old one before it writes (no other CTA reads it).
 *   2. recurrence + gated RMSNorm -> the out_proj activation.  One CTA per
 *      (sequence, V head), 256 threads, the 64 KB state in registers.
 *
 * Output: the out_proj input [rows][6144], emitted by the producer (rule 3)
 * as the A8 slot the dense arm reads -- E4M3 row-major, one ue8m0 per 32 at
 * pulsar_mx_sfoff(row, col/32, KBp) -- and/or as f32.  Either may be NULL,
 * not both; the E4M3 bytes encode exactly the f32 values.
 *
 * State layout, per slot (the caller owns the pools):
 *   conv state  f32 [3][10240]        inputs at positions -3, -2, -1
 *   rec state   f32 [48][128 k][128 v]
 * A fresh sequence starts from zeroed slots. */
#ifndef PULSAR_CUDA_GDN_H
#define PULSAR_CUDA_GDN_H

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
/* The engine's family ops (src/engine/family_qwen_s4.cpp) call the launcher
 * from a CUDA-free TU: the runtime's own typedef is all it needs (it passes 0,
 * the per-thread default stream the engine runs on).  Same guard as
 * pulsar_cuda_qwen.h. */
typedef struct CUstream_st *cudaStream_t;
#endif

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The model's shapes (config.json: linear_num_key_heads 16,
 * linear_num_value_heads 48, linear_key/value_head_dim 128,
 * linear_conv_kernel_dim 4, rms_norm_eps 1e-6).  The kernels are written for
 * exactly these; a different family instance is a different build. */
enum {
    PULSAR_GDN_NK       = 16,
    PULSAR_GDN_NV       = 48,
    PULSAR_GDN_DK       = 128,
    PULSAR_GDN_DV       = 128,
    PULSAR_GDN_KCONV    = 4,
    PULSAR_GDN_QKV_DIM  = 2 * PULSAR_GDN_NK * PULSAR_GDN_DK + PULSAR_GDN_NV * PULSAR_GDN_DV, /* 10240 */
    PULSAR_GDN_V_DIM    = PULSAR_GDN_NV * PULSAR_GDN_DV,                                      /* 6144  */
};

/** Conv-state floats per slot: 3 x 10240. */
#define PULSAR_GDN_CONV_STATE_FLOATS ((size_t)(PULSAR_GDN_KCONV - 1) * PULSAR_GDN_QKV_DIM)
/** Recurrent-state floats per slot: 48 x 128 x 128 (3 MiB). */
#define PULSAR_GDN_REC_STATE_FLOATS ((size_t)PULSAR_GDN_NV * PULSAR_GDN_DK * PULSAR_GDN_DV)

/** One layer's non-Linear parameters, all f32 on the device (the checkpoint's
 *  bf16 values widen exactly). */
typedef struct {
    /* bf16 -- the container's storage for these four (the graded recipe's `bf16`
     * class), widened in the kernels: a consumer reads the producer's format
     * (rule 3), so there is no f32 copy and no per-step conversion. */
    const uint16_t *conv_w;   /**< [10240][4], tap 3 multiplies the current token */
    const uint16_t *A_log;    /**< [48] */
    const uint16_t *dt_bias;  /**< [48] */
    const uint16_t *norm_w;   /**< [128], the gated norm's weight (used as is, no 1+w) */
} pulsar_gdn_weights;

/** One call: n_seq sequences of seq_rows consecutive rows each (rows =
 *  n_seq * seq_rows) -- the family step's two modes: DECODE is n_seq rows of
 *  one token (seq_rows 1), a PREFILL chunk is one sequence (n_seq 1) of
 *  seq_rows tokens.  row_slot is per ROW (the step's row_bank slot as it is);
 *  sequence s runs on slot row_slot[s * seq_rows].  Pointers are device
 *  pointers; every f32 pointer 16-byte aligned and every pitch a multiple of
 *  4 floats. */
typedef struct {
    int            n_seq;
    int            seq_rows;
    const int32_t *row_slot;      /**< [n_seq * seq_rows] state slot (bank) of each row */
    float         *conv_state;    /**< pool, PULSAR_GDN_CONV_STATE_FLOATS per slot */
    float         *rec_state;     /**< pool, PULSAR_GDN_REC_STATE_FLOATS per slot */
    const float   *qkv; int ld_qkv;   /**< in_proj_qkv output [rows][10240] */
    const float   *z;   int ld_z;     /**< in_proj_z output [rows][6144] */
    const float   *a;   int ld_a;     /**< in_proj_a output [rows][48] */
    const float   *b;   int ld_b;     /**< in_proj_b output [rows][48] */
    void          *scratch;           /**< >= pulsar_gdn_scratch_bytes(rows) */
    size_t         scratch_bytes;
    float         *out_f32;           /**< [rows][6144] or NULL */
    void          *out_e4m3;          /**< A8 slot data [rows][6144] or NULL */
    void          *out_scale;         /**< A8 slot ue8m0 scales (zeroed by the slot owner) */
    int            out_kbp;           /**< must equal pulsar_mx_kbp(6144) = 192 */
} pulsar_gdn_call;

/** Scratch bytes a call of up to `rows` rows needs: f32 q^ k^ v [rows][10240] and
 *  decay, beta [rows][96]. */
size_t pulsar_gdn_scratch_bytes(int rows);

/** Run the whole non-Linear GDN block for one layer on `stream` (two
 *  kernels).  Advances both states of every sequence in the call.
 *  Distinct sequences of one call must be on distinct slots.
 *  @return 0 on success, -1 on a refused contract (named on stderr, nothing
 *  launched), -3 on a launch failure. */
int pulsar_gdn_forward(const pulsar_gdn_weights *w, const pulsar_gdn_call *c, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* PULSAR_CUDA_GDN_H */

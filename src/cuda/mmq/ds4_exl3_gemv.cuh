// SPDX-License-Identifier: MIT
// The EXL3 routed-expert arm (L245): a k-major trellis GEMV over the
// per-expert [trellis | scales] slices, plus the two EXL3 epilogues that carry
// the Hadamard rotations the format puts on both sides of every projection.
//
// Contracts mirror the IQ2 D2R launchers (ds4_mmq_d2r.cuh): the caller has
// sorted the (token, slot) pairs by expert (ids_dst, expert_bounds) and staged
// the producer's E4M3 activation as block_mx_act_mmq [K/128][n_assign].  The
// tables are exl3_expert_table()'s interleaved [trellis, scales] pointer pairs.
// Every row takes this arm -- decode and prefill alike -- so there is no
// decode/prefill numerics boundary in the EXL3 lane.

#pragma once

#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

#include "engine/exl3_trellis.h"   /* EXL3_ARM_*: the kinds and their rates */

/** The GEMV kinds are exl3_trellis.h's arms: EXL3_ARM_DOWN (one projection
 *  whose input the fold already rotated), EXL3_ARM_PAIR (gate and up from two
 *  slices -- DeepSeek's split stacks -- each rotating the input by its own suh),
 *  EXL3_ARM_GATE_UP_FUSED (one [in -> 2 mid] slice whose output rows are
 *  gate | up -- Qwen3.8-Flash-Next, L251 -- rotating the input by its suh).
 *  True when the kind instantiates rate `k2` (exl3_arm_has_rate, the one table). */
bool ds4_exl3_gemv_rate_supported(int kind, int k2);

/** gate/up: out_gate / out_up [n_assign][M] f32 = the UNROTATED z of each
 *  assignment (the fold applies svh and the output Hadamard).  The input
 *  rotation x * suh -> H128 is done in-kernel, once per (assignment,
 *  projection), because suh is per expert-projection.  M % 32 == 0,
 *  K % 128 == 0, K <= 5120. */
int ds4_exl3_moe_gemv_pair_launch(
    const void    * gate_table,
    const void    * up_table,
    int             k2,
    const void    * act,
    const int32_t * ids_dst,
    const int32_t * expert_bounds,
    float         * out_gate,
    float         * out_up,
    int             M,
    int             K,
    int64_t         n_assign,
    int             n_experts,
    cudaStream_t    stream);

/** The pair GEMV with the row block pinned (1, 4 or 16 assignments per CTA)
 *  instead of chosen from n_assign.  Every row block is bit-identical; the
 *  gate uses this to prove it, the arm never calls it. */
int ds4_exl3_moe_gemv_pair_launch_rows(
    const void    * gate_table,
    const void    * up_table,
    int             k2,
    const void    * act,
    const int32_t * ids_dst,
    const int32_t * expert_bounds,
    float         * out_gate,
    float         * out_up,
    int             M,
    int             K,
    int64_t         n_assign,
    int             n_experts,
    int             rows_per_block,
    cudaStream_t    stream);

/** fused gate_up: out [n_assign][M] f32 = the UNROTATED z of the one [K -> M]
 *  slice (M = 2 mid: gate rows then up rows), the input rotated in-kernel by
 *  the slice's suh.  Same contract as the pair launch otherwise. */
/* L251 / ac69748f: the same fused gate_up arm over a BF16 activation.  The Qwen family has no E4M3
 * activation slot, so its routed experts read row-major bf16. */
int ds4_exl3_moe_gemv_fused_bf16_launch(
        const void *table, int k2, const void *act,
        const int32_t *ids_dst, const int32_t *ids_src, const int32_t *expert_bounds,
        float *out, int M, int K, int64_t n_assign,
        int n_experts, cudaStream_t stream);

/* The down arm over the same bf16 activation (its input is the fold's output). */
int ds4_exl3_moe_gemv_single_bf16_launch(
        const void *table, int k2, const void *act,
        const int32_t *ids_dst, const int32_t *ids_src, const int32_t *expert_bounds,
        float *out, int M, int K, int64_t n_assign,
        int n_experts, cudaStream_t stream);

/* L251: the Qwen routed arm for PROMPT CHUNKS (qwen_exl3_moe_prefill.cu) -- a grouped tensor-core GEMM
 * over the same expert-sorted schedule, writing the same unrotated z as the two bf16 GEMVs above.
 * rotate_input: the fused gate_up (x * suh, H128); false: the down (input rotated by the fold).
 * Rates k2 = 8, 10; M % 64, K % 128.  Agrees with the GEMV to rounding, not to the bit. */
constexpr int64_t QWEN_EXL3_MOE_PREFILL_MIN_ASSIGN = 161;   /* more than 16 tokens x 10 slots */
int qwen_exl3_moe_prefill_launch(
        const void *table, int k2, bool rotate_input, const void *x_bf16,
        const int32_t *ids_dst, const int32_t *ids_src, const int32_t *expert_bounds,
        float *out, int M, int K, int64_t n_assign, int n_experts, cudaStream_t stream);

int ds4_exl3_moe_gemv_fused_launch(
    const void    * table,
    int             k2,
    const void    * act,
    const int32_t * ids_dst,
    const int32_t * expert_bounds,
    float         * out,
    int             M,
    int             K,
    int64_t         n_assign,
    int             n_experts,
    cudaStream_t    stream);

/** The fused launch with the row block pinned (1, 4 or 16); the gate uses it
 *  to prove every row block bit-identical, the arm never calls it. */
int ds4_exl3_moe_gemv_fused_launch_rows(
    const void    * table,
    int             k2,
    const void    * act,
    const int32_t * ids_dst,
    const int32_t * expert_bounds,
    float         * out,
    int             M,
    int             K,
    int64_t         n_assign,
    int             n_experts,
    int             rows_per_block,
    cudaStream_t    stream);

/** down: out [n_assign][M] f32 = the UNROTATED z_d; the input (mid) arrives
 *  pre-rotated by the fold, so nothing is rotated here. */
int ds4_exl3_moe_gemv_single_launch(
    const void    * table,
    int             k2,
    const void    * act,
    const int32_t * ids_dst,
    const int32_t * expert_bounds,
    float         * out,
    int             M,
    int             K,
    int64_t         n_assign,
    int             n_experts,
    cudaStream_t    stream);

/** The EXL3 SwiGLU fold: per pair (pair = tok * n_expert + slot, expert =
 *  selected[pair]) and per 128-block of mid: y_g = svh_g * H(z_g),
 *  y_u = svh_u * H(z_u), v = swiglu(y_g, y_u) * weight, then the DOWN input
 *  rotation t = H(v * suh_d), emitted as E4M3 in 32-groups into the mid slot
 *  (mid_q, mid_sf, mid_kbp) -- the producer rule for the down projection.
 *  mid_dim % 128 == 0. */
int ds4_exl3_moe_fold_launch(
    const float   * gate_z,
    const float   * up_z,
    const int32_t * selected,
    const float   * weights,
    const void    * gate_table,
    const void    * up_table,
    const void    * down_table,
    int             in_dim,
    int             mid_dim,
    int64_t         pairs,
    float           clamp,
    void          * mid_q,
    void          * mid_sf,
    int             mid_kbp,
    cudaStream_t    stream,
    void          * mid_bf16 = nullptr);

/** The fold over a FUSED gate_up: gate_up_z [pairs][2 mid] (gate columns
 *  0..mid-1, up columns mid..2 mid-1), svh of the one gate_up slice (gate's at
 *  suh + in_dim, up's at suh + in_dim + mid); otherwise the same arithmetic and
 *  the same E4M3 mid as ds4_exl3_moe_fold_launch. */
int ds4_exl3_moe_fold_fused_launch(
    const float   * gate_up_z,
    const int32_t * selected,
    const float   * weights,
    const void    * gate_up_table,
    const void    * down_table,
    int             in_dim,
    int             mid_dim,
    int64_t         pairs,
    float           clamp,
    void          * mid_q,
    void          * mid_sf,
    int             mid_kbp,
    cudaStream_t    stream,
    void          * mid_bf16 = nullptr);

/** The EXL3 sum: out[tok][o] = sum over slots (in slot order) of
 *  svh_d * H(z_d[pair]) -- the down projection's output rotation folded into
 *  the fixed-order reduction.  A non-finite sum records nf_code in *nf_flag
 *  (first writer wins), the L188 contract.  out_dim % 128 == 0. */
int ds4_exl3_moe_sum_launch(
    float         * out,
    const float   * down_z,
    const int32_t * selected,
    const void    * down_table,
    int             mid_dim,
    int             out_dim,
    int             n_expert,
    int             n_tokens,
    uint32_t      * nf_flag,
    uint32_t        nf_code,
    cudaStream_t    stream);

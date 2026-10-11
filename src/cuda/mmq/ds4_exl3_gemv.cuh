// SPDX-License-Identifier: MIT
// The EXL3 routed-expert arm (L245): a k-major trellis GEMV over the
// per-expert [trellis | scales] slices, plus the two EXL3 epilogues that carry
// the Hadamard rotations the format puts on both sides of every projection.
//
// The caller has sorted the (token, slot) pairs by expert (ids_dst, ids_src,
// expert_bounds: mm_ids_helper's schedule) and hands the activation as its
// producer emitted it (exl3_moe_act: bf16 rows or the E4M3 slot).  The tables
// are exl3_expert_table()'s interleaved [trellis, scales] pointer pairs.
// Decode rows take these GEMVs; prompt chunks take the tensor-core prefill arm
// (exl3_moe_prefill.cuh, exl3_moe_prefill_takes is the rule) -- the two agree
// to rounding, and a row's bits never depend on its prompt chunk.  The routed
// MoE's one front door, which composes them, is exl3_moe.cuh.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

#include "engine/exl3_trellis.h"   /* EXL3_ARM_*: the kinds and their rates */
#include "exl3_moe_prefill.cuh"     /* exl3_moe_act: the activation as its producer emitted it */

/** The GEMV kinds are exl3_trellis.h's arms: EXL3_ARM_DOWN (one projection
 *  whose input the fold already rotated), EXL3_ARM_PAIR (gate and up from two
 *  slices -- DeepSeek's split stacks -- each rotating the input by its own suh),
 *  EXL3_ARM_GATE_UP_FUSED (one [in -> 2 mid] slice whose output rows are
 *  gate | up -- Qwen3.8-Flash-Next, L251 -- rotating the input by its suh).
 *  True when the kind instantiates rate `k2` (exl3_arm_has_rate, the one table). */
bool ds4_exl3_gemv_rate_supported(int kind, int k2);

/** One routed GEMV over the expert-sorted schedule: out[ids_dst[c]][M] f32 =
 *  the UNROTATED z of assignment c (the fold / sum apply svh and the output
 *  Hadamard), its input row ids_src[c] of `act`.  kind EXL3_ARM_PAIR writes
 *  gate to `out` and up to `out_up` from `table` / `up_table`, each rotating
 *  the input by its own suh in-kernel (suh sits inside the Hadamard and is
 *  per expert-projection, so it cannot be hoisted before routing);
 *  EXL3_ARM_GATE_UP_FUSED rotates by the one slice's suh (output rows
 *  gate | up); EXL3_ARM_DOWN reads an input the fold already rotated.
 *  Assignments past expert_bounds[n_experts] are not computed.  M % 32 == 0,
 *  K % 128 == 0, K <= 5120.  rows_per_block 0 = the arm's choice by n_assign;
 *  1, 4 or 16 pins it (every row block is bit-identical -- the gate proves it). */
int ds4_exl3_moe_gemv_launch(
    int                  kind,
    const void         * table,
    const void         * up_table,
    int                  k2,
    const exl3_moe_act & act,
    const int32_t      * ids_dst,
    const int32_t      * ids_src,
    const int32_t      * expert_bounds,
    float              * out,
    float              * out_up,
    int                  M,
    int                  K,
    int64_t              n_assign,
    int                  n_experts,
    int                  rows_per_block,
    cudaStream_t         stream);

/* The routed arm for PROMPT CHUNKS, every family, is exl3_moe_prefill.cuh: a grouped tensor-core GEMM
 * over the same expert-sorted schedule, writing the same unrotated z as the GEMVs here. */
/* L251: the dense arm's prompt chunks through a decode-once tensor-core GEMM: w is the
 * dense layout (trellis | suh[K] | svh[N]); writes y [M][N] finished (output rotation and svh
 * applied).  ds4_exl3_dense_launch calls it for M > 16. */
int qwen_exl3_dense_prefill_launch(const void *w, int k2, const void *x_bf16, float *y, int M, int K, int N,
                                   uint64_t trellis_bytes, cudaStream_t stream);

/** A shared expert riding the routed launches (exl3_moe.cuh): its SwiGLU h = silu(g) * u over [rows][mid] f32
 *  planes, emitted bf16 for its down, in the fold's launch.  g, u, h are 16-byte aligned. */
struct exl3_moe_shared_swiglu {
    const float   * g;
    const float   * u;
    __nv_bfloat16 * h;
    int             rows, mid;
};

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
    void          * mid_bf16 = nullptr,
    const exl3_moe_shared_swiglu * shared = nullptr);

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
    void          * mid_bf16 = nullptr,
    const exl3_moe_shared_swiglu * shared = nullptr);

/** The EXL3 sum: out[tok][o] = sum over slots (in slot order) of
 *  svh_d * H(z_d[pair]) -- the down projection's output rotation folded into
 *  the fixed-order reduction.  A non-finite sum records nf_code in *nf_flag
 *  (first writer wins), the L188 contract.  out_dim % 128 == 0.  A riding shared
 *  expert's output is added after the routed sum (the flag sees the routed sum
 *  alone): out = sum + shared_scale[tok] * shared_y (scale 1 when NULL). */
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
    cudaStream_t    stream,
    const float   * shared_y = nullptr,
    const float   * shared_scale = nullptr);

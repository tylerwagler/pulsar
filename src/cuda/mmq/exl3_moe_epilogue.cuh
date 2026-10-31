// SPDX-License-Identifier: MIT
/** @file
 *  Contracts of the EXL3 routed MoE's fold and sum epilogues. */
// The EXL3 routed MoE's two epilogues -- the fold and the sum -- which carry the Hadamard rotations the format
// puts on both sides of every projection.  The projections are the tensor-core arms (exl3_moe_prefill.cuh: the
// decode arm for decode rows, the grouped GEMM for prompt chunks), which write the UNROTATED z of each
// assignment over the expert-sorted schedule; a decode launch leaves it as K-split partials (exl3_moe_partials)
// that these kernels add in split order.  The tables are exl3_expert_table()'s interleaved [trellis, scales]
// pointer pairs.  The routed MoE's one front door, which composes them, is exl3_moe.cuh.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

#include "engine/exl3_trellis.h"   /* the scales layout */
#include "exl3_moe_prefill.cuh"     /* exl3_moe_partials */
#include "cuda/pulsar_cuda_router.h" /* PULSAR_ROUTER_SHARED_MAX */

/** The shared experts' outputs the sum adds after the routed sum, in order: out += scale[j][tok] * y[j] (scale 1
 *  when scale[j] is NULL), j = 0 .. n-1. */
struct exl3_moe_shared_add {
    /// Shared expert j's output, f32 [tokens][out_dim], device (16-byte aligned rows).
    const float *y[PULSAR_ROUTER_SHARED_MAX];
    /// Shared expert j's per-token scale f32 [tokens], or NULL for 1.
    const float *scale[PULSAR_ROUTER_SHARED_MAX];
    /// Shared experts riding the sum, 0 .. PULSAR_ROUTER_SHARED_MAX.
    int n;
};

/** The EXL3 SwiGLU fold: per pair (pair = tok * n_expert + slot, expert =
 *  selected[pair]) and per 128-block of mid: y_g = svh_g * H(z_g),
 *  y_u = svh_u * H(z_u), v = swiglu(y_g, y_u) * weight, then the DOWN input
 *  rotation t = H(v * suh_d), emitted as E4M3 in 32-groups into the mid slot
 *  (mid_q, mid_sf, mid_kbp) -- the producer rule for the down projection.
 *  mid_dim % 128 == 0.  z_splits > 1: gate_z / up_z are a decode launch's K-split partials (exl3_moe_partials),
 *  z_split_stride floats apart, added in split order before anything else. */
/// @param gate_z the gate projection's unrotated z, f32 [pairs][mid_dim] (or its K-split partials).
/// @param up_z the up projection's unrotated z, f32 [pairs][mid_dim] (or its K-split partials).
/// @param selected the routes, i32 [tokens][n_expert] (< 0: no expert).
/// @param weights the route weights, f32 [pairs].
/// @param gate_table the gate stack's exl3_expert_table pairs.
/// @param up_table the up stack's exl3_expert_table pairs.
/// @param down_table the down stack's exl3_expert_table pairs.
/// @param in_dim the block input width.
/// @param mid_dim the expert intermediate width (a multiple of 128).
/// @param pairs routed pairs (tokens x n_expert).
/// @param clamp the SwiGLU clamp.
/// @param mid_q out: the down input's E4M3 slot data, [pairs][mid_dim].
/// @param mid_sf out: its ue8m0 plane (pulsar_mx_sfoff).
/// @param mid_kbp the plane's blocks per row.
/// @param stream the stream.
/// @param mid_bf16 out: the down input as bf16 rows instead of the slot, or NULL.
/// @param z_splits K-split partial planes in the z input (1: a finished z).
/// @param z_split_stride floats between partial planes.
/// @return 0, -1 refused, -3 launch failure.
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
    int             z_splits = 1,
    int64_t         z_split_stride = 0);

/** The fold over a FUSED gate_up: gate_up_z [pairs][2 mid] (gate columns
 *  0..mid-1, up columns mid..2 mid-1), svh of the one gate_up slice (gate's at
 *  suh + in_dim, up's at suh + in_dim + mid); otherwise the same arithmetic and
 *  the same E4M3 mid as ds4_exl3_moe_fold_launch.  The K-split partials as ds4_exl3_moe_fold_launch.*/
/// @param gate_up_z the fused gate_up z, f32 [pairs][2 mid_dim] (or its K-split partials).
/// @param selected the routes, i32 [tokens][n_expert] (< 0: no expert).
/// @param weights the route weights, f32 [pairs].
/// @param gate_up_table the fused gate_up stack's exl3_expert_table pairs.
/// @param down_table the down stack's exl3_expert_table pairs.
/// @param in_dim the block input width.
/// @param mid_dim the expert intermediate width (a multiple of 128).
/// @param pairs routed pairs (tokens x n_expert).
/// @param clamp the SwiGLU clamp.
/// @param mid_q out: the down input's E4M3 slot data, [pairs][mid_dim].
/// @param mid_sf out: its ue8m0 plane (pulsar_mx_sfoff).
/// @param mid_kbp the plane's blocks per row.
/// @param stream the stream.
/// @param mid_bf16 out: the down input as bf16 rows instead of the slot, or NULL.
/// @param z_splits K-split partial planes in the z input (1: a finished z).
/// @param z_split_stride floats between partial planes.
/// @return 0, -1 refused, -3 launch failure.
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
    int             z_splits = 1,
    int64_t         z_split_stride = 0);

/** L296: the UP-ONLY fold (an ungated expert: no gate projection): y_u = svh_u * H(z_u), v = relu(y_u)^2 * weight
 *  (pulsar_relu2_elem), then the down input rotation and the emit exactly as ds4_exl3_moe_fold_launch -- the same
 *  kernel with the gate side compiled out.  up_z [pairs][mid] (the up slice's svh at suh + in_dim); no clamp.  The
 *  K-split partials as ds4_exl3_moe_fold_launch. */
/// @param up_z the up projection's unrotated z, f32 [pairs][mid_dim] (or its K-split partials).
/// @param selected the routes, i32 [tokens][n_expert] (< 0: no expert).
/// @param weights the route weights, f32 [pairs].
/// @param up_table the up stack's exl3_expert_table pairs.
/// @param down_table the down stack's exl3_expert_table pairs.
/// @param in_dim the block input width.
/// @param mid_dim the expert intermediate width (a multiple of 128).
/// @param pairs routed pairs (tokens x n_expert).
/// @param mid_q out: the down input's E4M3 slot data, [pairs][mid_dim].
/// @param mid_sf out: its ue8m0 plane (pulsar_mx_sfoff).
/// @param mid_kbp the plane's blocks per row.
/// @param stream the stream.
/// @param mid_bf16 out: the down input as bf16 rows instead of the slot, or NULL.
/// @param z_splits K-split partial planes in the z input (1: a finished z).
/// @param z_split_stride floats between partial planes.
/// @return 0, -1 refused, -3 launch failure.
int ds4_exl3_moe_fold_up_launch(
    const float   * up_z,
    const int32_t * selected,
    const float   * weights,
    const void    * up_table,
    const void    * down_table,
    int             in_dim,
    int             mid_dim,
    int64_t         pairs,
    void          * mid_q,
    void          * mid_sf,
    int             mid_kbp,
    cudaStream_t    stream,
    void          * mid_bf16 = nullptr,
    int             z_splits = 1,
    int64_t         z_split_stride = 0);

/** The EXL3 sum: out[tok][o] = sum over slots (in slot order) of
 *  svh_d * H(z_d[pair]) -- the down projection's output rotation folded into
 *  the fixed-order reduction.  A non-finite sum records nf_code in *nf_flag
 *  (first writer wins), the L188 contract.  out_dim % 128 == 0.  Riding shared
 *  experts' outputs are added after the routed sum (the flag sees the routed sum
 *  alone): out = sum + scale_0[tok] * y_0 + scale_1[tok] * y_1 + ... (`shared`, NULL = none).  z_splits > 1:
 *  down_z is a decode launch's K-split partials, z_split_stride floats apart, added in split order first. */
/// @param out out: f32 [n_tokens][out_dim].
/// @param down_z the down projection's unrotated z, f32 [pairs][out_dim] (or its K-split partials).
/// @param selected the routes, i32 [tokens][n_expert] (< 0: no expert).
/// @param down_table the down stack's exl3_expert_table pairs.
/// @param mid_dim the expert intermediate width (a multiple of 128).
/// @param out_dim the block output width (a multiple of 128).
/// @param n_expert experts per token (slots).
/// @param n_tokens tokens.
/// @param nf_flag the non-finite flag (first writer wins).
/// @param nf_code the code written to it.
/// @param stream the stream.
/// @param shared riding shared experts added after the routed sum, or NULL.
/// @param z_splits K-split partial planes in the z input (1: a finished z).
/// @param z_split_stride floats between partial planes.
/// @return 0, -1 refused, -3 launch failure.
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
    const exl3_moe_shared_add * shared = nullptr,
    int             z_splits = 1,
    int64_t         z_split_stride = 0);

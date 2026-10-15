// SPDX-License-Identifier: MIT
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
    const float *y[PULSAR_ROUTER_SHARED_MAX];
    const float *scale[PULSAR_ROUTER_SHARED_MAX];
    int n;
};

/** The EXL3 SwiGLU fold: per pair (pair = tok * n_expert + slot, expert =
 *  selected[pair]) and per 128-block of mid: y_g = svh_g * H(z_g),
 *  y_u = svh_u * H(z_u), v = swiglu(y_g, y_u) * weight, then the DOWN input
 *  rotation t = H(v * suh_d), emitted as E4M3 in 32-groups into the mid slot
 *  (mid_q, mid_sf, mid_kbp) -- the producer rule for the down projection.
 *  mid_dim % 128 == 0.  z_splits > 1: gate_z / up_z are a decode launch's K-split partials (exl3_moe_partials),
 *  z_split_stride floats apart, added in split order before anything else. */
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

/** The EXL3 sum: out[tok][o] = sum over slots (in slot order) of
 *  svh_d * H(z_d[pair]) -- the down projection's output rotation folded into
 *  the fixed-order reduction.  A non-finite sum records nf_code in *nf_flag
 *  (first writer wins), the L188 contract.  out_dim % 128 == 0.  Riding shared
 *  experts' outputs are added after the routed sum (the flag sees the routed sum
 *  alone): out = sum + scale_0[tok] * y_0 + scale_1[tok] * y_1 + ... (`shared`, NULL = none).  z_splits > 1:
 *  down_z is a decode launch's K-split partials, z_split_stride floats apart, added in split order first. */
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

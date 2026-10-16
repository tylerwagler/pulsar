// SPDX-License-Identifier: MIT
// L295 F7: BF16 WEIGHTS against raw bf16 ROWS -- the dense Linear and the routed experts, every family.
//
// A family that keeps a Linear, its shared experts or its routed experts in BF16 (a BF16 grading container; plan
// 120's BF16 shared-expert fallback) hands the rows its producer emitted (rule 3), as the EXL3 and mxfp8_lt rows arms
// take them.  Admission is weight_format.cpp's (PULSAR_DENSE_ARM_BF16_ROWS; the expert roles at bf16 rows); the
// launchers are the rows front doors' (pulsar_rows_linear_launch, pulsar_rows_moe_routed_launch).
//
//   dense, decode rows (not a prompt chunk): the tensor-core ROWS kernel -- a warp owns 16 weight rows (the MMA's
//       M), up to PULSAR_BF16_ROWS_DECODE_MAX of the call's rows are its N (wider calls run blocks of that many in
//       turn), bf16 x bf16 products exact in f32, a fresh 16-k fragment added into the row's accumulator in k order.
//       A row's column reads only its own activation, so its bytes do not depend on how many rows ride with it
//       (M-independent at every width, as the EXL3 and mxfp8 decode arms are).
//   dense, prompt chunks: cuBLASLt with ONE kernel per weight shape, chosen at the canonical prefill row count,
//       split-K masked out (pulsar_gpu_bf16_rows_gemm, the L183 contract): a prompt row's bytes do not depend on its
//       chunk.  It agrees with the rows kernel to rounding, not to the bit (as the other arms' prompt GEMMs do).
//   routed (gate + up pair and a down, each [n_expert][out][in] bf16): the same rows kernel over the expert-sorted
//       schedule (mm_ids_helper), up to 64 of an expert's assignments a work item -- every width and row kind, so a
//       prompt row and a decode row are the same bytes; the SwiGLU with the route weight folded in, rounded to the
//       bf16 rows the down reads (the EXL3 fold's expression); the down; the slot-ordered sum.
#pragma once

#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

#include "pulsar_cuda_qwen.h"   /* pulsar_ffn_act: the rows front door's neutral activation enum */

/** The rows one launch of the M-independent rows kernel takes (wider decode calls run blocks of it). */
#define PULSAR_BF16_ROWS_DECODE_MAX 16

/** y [rows][out] f32 = W x for a bf16 weight W [out][in] (row-major, the container's [in, out] ne order) and bf16
 *  rows x [rows][in].  w, x and y 16-byte aligned, in % 64 == 0.  `prompt`: a prompt chunk (the fixed-order GEMM at
 *  every width).  Refuses (-1) on a contract violation. */
int pulsar_bf16_rows_linear_launch(const void *w, const uint16_t *x_bf16, int rows, int in, int out, float *y,
                                   bool prompt, cudaStream_t stream);

/** The fixed-order cuBLASLt bf16 GEMM over raw bf16 rows (pulsar_cuda_matmul.cu): y [rows][out] f32.  1 on success. */
int pulsar_gpu_bf16_rows_gemm(float *y, const uint16_t *w, const uint16_t *x_bf16, uint64_t in_dim, uint64_t out_dim,
                              uint64_t rows, cudaStream_t stream);

/** One layer's routed BF16 stacks over the experts this rank holds: gate / up [n_experts][mid][in], down
 *  [n_experts][out][in = mid], bf16, 16-byte aligned.  `act` the activation (pulsar_cuda_qwen.h pulsar_ffn_act):
 *  SWIGLU reads gate and up, RELU2 up alone (gate NULL -- an up-only, non-gated expert). */
struct pulsar_bf16_moe {
    const void *gate, *up, *down;
    int n_experts;
    int in_dim, mid_dim, out_dim;
    pulsar_ffn_act act;
};

/** out [T][out_dim] f32 = sum over each row's top_k picks (sel [T][top_k], local expert ids; < 0 is another rank's
 *  or the router's NaN route and adds nothing) of down(bf16(act(...) * w)), slots in order: act swiglu(gate x, up x)
 *  or relu(up x)^2, the route weight folded in before the one bf16 rounding (as the EXL3 fold does).  The
 *  caller's buffers: gate_z (unused for RELU2) / up_z [T * top_k][mid] f32, mid [T * top_k][mid] bf16, down_z
 *  [T * top_k][out] f32.  A
 *  non-finite output records nf_code in *nf_flag (first writer wins).  Needs the MMQ drivers' schedule
 *  (PULSAR_HAVE_MMQ).  Refuses (-1) on a contract violation. */
int pulsar_bf16_moe_routed_launch(const pulsar_bf16_moe &w, const int32_t *sel, const float *wts, int T, int top_k,
                                  float clamp, const uint16_t *x_bf16, float *gate_z, float *up_z, uint16_t *mid,
                                  float *down_z, float *out, uint32_t *nf_flag, uint32_t nf_code, cudaStream_t stream);

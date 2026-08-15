// SPDX-License-Identifier: MIT
// Internal launcher for the gated D2R Q2_K MoE down-GEMM path.

#pragma once

#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

/* Which IQ2 dequant arm the expert GEMMs run; read once per process.
 *   0 = q8_1 int8 operands, integer MMA, scale fold (default)
 *   1 = E4M3 operands, block-scaled MXFP8 MMA, hardware ue8m0, no fold
 * ⚠ The INPUT STAGING must match: arm 1 requires ds4_quantize_mmq_e4m3_cuda.
 * They are selected off this same function in ds4_mmq.cu; keep them together. */

/* Stage MMQ activations as E4M3 + ue8m0 instead of q8_1 int8.  Same
 * block_q8_1_mmq layout: qs[] holds e4m3 bit patterns, d4[i] carries the ue8m0
 * byte as a float. */
void ds4_quantize_mmq_e4m3_cuda(
        const float *x, const int32_t *ids, void *vy,
        int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
        int n_expert_used, bool scatter, cudaStream_t stream);

bool ds4_mmq_iq2_xxs_moe_d2r_available(int cc);

size_t ds4_mmq_iq2_xxs_moe_d2r_pair_scratch_bytes(int64_t ncols_max, int n_experts);


int ds4_mmq_iq2_xxs_moe_d2r_pair_launch(
    const void    * gate_soa,
    const void    * up_soa,
    int64_t         soa_blocks,
    const void    * q8,
    const int32_t * ids_dst,
    const int32_t * expert_bounds,
    float         * out_gate,
    float         * out_up,
    int             M,
    int             K,
    int64_t         ne_get_rows,
    int             n_experts,
    void          * worklist_scratch,
    size_t          worklist_scratch_bytes,
    cudaStream_t    stream);

// Complete target-prefill gate/up path: both IQ2_XXS projections share one
// activation tile, then sanitize + clamp + SwiGLU + routing weight are folded
// directly into the expert-major Q8_1 D2S6 input consumed by Q2_K down.

int ds4_mmq_iq2_xxs_moe_d2r_single_launch(const void *W_soa, int64_t soa_blocks, const void *q8, const int32_t *ids_dst, const int32_t *expert_bounds, float *out, int M, int K, int64_t ne_get_rows, int n_experts, void *worklist_scratch, size_t worklist_scratch_bytes, cudaStream_t stream);

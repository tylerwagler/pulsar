// SPDX-License-Identifier: MIT
// Internal launchers for the D2R IQ2_XXS MoE GEMMs (pair gate/up + single)
// and their E4M3 activation staging.  Ungated: the E4M3 arm is the only arm.

#pragma once

#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

/* Stage MMQ activations in the block_mx_act_mmq layout (qs[] = E4M3 bit
 * patterns, d4[i] = the block's ue8m0 byte as a float), copied from the
 * encoding the producing norm already made: the activation cache's [row][ne00]
 * E4M3 plus a ue8m0 plane in the pulsar_mx_sfoff swizzle.  A copy, not a
 * re-encode.  src_kbp is the cache's blocks-per-row pitch. */
void ds4_gather_mmq_e4m3_cuda(
        const void *src_q, const void *src_sf, int src_kbp,
        const int32_t *ids, void *vy,
        int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
        int n_expert_used, bool scatter, cudaStream_t stream);

bool ds4_mmq_iq2_xxs_moe_d2r_available(int cc);

size_t ds4_mmq_iq2_xxs_moe_d2r_pair_scratch_bytes(int64_t ncols_max, int n_experts);


int ds4_mmq_iq2_xxs_moe_d2r_pair_launch(
    const void    * gate_soa,
    const void    * up_soa,
    int64_t         soa_blocks,
    const void    * act,
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

// Single-tensor IQ2_XXS D2R GEMM (the routed DOWN leg): one weight, one
// f32 output, same worklist/staging contract as the pair launch above.
// (A comment here used to describe a deleted fused Q8_1/Q2_K epilogue.)

int ds4_mmq_iq2_xxs_moe_d2r_single_launch(const void *W_soa, int64_t soa_blocks, const void *act, const int32_t *ids_dst, const int32_t *expert_bounds, float *out, int M, int K, int64_t ne_get_rows, int n_experts, void *worklist_scratch, size_t worklist_scratch_bytes, cudaStream_t stream);

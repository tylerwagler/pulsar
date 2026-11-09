// SPDX-License-Identifier: MIT
/** @file
 *  Internal launchers of the D2R IQ2_XXS MoE GEMMs and their E4M3 activation staging. */
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
/// Launches the gather (ds4_quantize_e4m3.cu).
/// @param src_q the cache's E4M3 rows, [rows][ne00].
/// @param src_sf their ue8m0 plane (pulsar_mx_sfoff).
/// @param src_kbp the plane's blocks per row.
/// @param ids the routes, i32 [n_tokens][n_expert_used].
/// @param vy out: the block_mx_act_mmq staging.
/// @param ne00 source row width (a multiple of 4).
/// @param s01 unused (upstream stride).
/// @param s02 unused (upstream stride).
/// @param s03 unused (upstream stride).
/// @param ne0 staged row width (a multiple of 128; zero-filled past ne00).
/// @param ne1 staged columns (assignments).
/// @param ne2 z slices (1 in every live launch).
/// @param ne3 outer slices (1).
/// @param n_expert_used experts per token.
/// @param scatter one source row per token written to each of its n_expert_used slots (else a gather by ids).
/// @param stream the stream.
void ds4_gather_mmq_e4m3_cuda(
        const void *src_q, const void *src_sf, int src_kbp,
        const int32_t *ids, void *vy,
        int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
        int n_expert_used, bool scatter, cudaStream_t stream);

/// @param cc the device's compute capability, major * 100 + minor * 10 (ggml_cuda_info).
/// @return true when the device has the block-scaled MMA and the vendored grid holds only {8, 25, 43} (cached per cc).
bool ds4_mmq_iq2_xxs_moe_d2r_available(int cc);

/// @param ncols_max the most assignment columns a call will carry.
/// @param n_experts experts in the stack (0 for a dense matmul).
/// @return work-list bytes, or 0 for a shape past the 32-bit capacity.
size_t ds4_mmq_iq2_xxs_moe_d2r_pair_scratch_bytes(int64_t ncols_max, int n_experts);


/// The gate + up D2R GEMM over the expert-sorted schedule (the decode GEMV tier at decode widths).
/// @param gate_soa the gate weight, aligned-SoA IQ2_XXS.
/// @param up_soa the up weight, same shape.
/// @param soa_blocks super-blocks in the artifact (n_experts * M * K / 256), checked against the shape.
/// @param act the staged activation (block_mx_act_mmq, expert-major).
/// @param ids_dst per assignment, its output column.
/// @param expert_bounds per expert, its first assignment, [n_experts + 1].
/// @param out_gate out: the gate output.
/// @param out_up out: the up output.
/// @param M weight rows (output width).
/// @param K input width (a multiple of 256).
/// @param ne_get_rows assignment slots (tokens x top-k).
/// @param n_experts experts in the stack (0 for a dense matmul).
/// @param worklist_scratch device scratch for the work list.
/// @param worklist_scratch_bytes bytes of it (ds4_mmq_iq2_xxs_moe_d2r_pair_scratch_bytes).
/// @param stream the stream.
/// @return 0; -1 refused (including no D2R arm on this device, said); -2 / -3 launch failures.
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

/// The single-weight D2R GEMM (the routed down).
/// @param W_soa the weight, aligned-SoA IQ2_XXS.
/// @param soa_blocks super-blocks in the artifact (n_experts * M * K / 256), checked against the shape.
/// @param act the staged activation (block_mx_act_mmq, expert-major).
/// @param ids_dst per assignment, its output column.
/// @param expert_bounds per expert, its first assignment, [n_experts + 1].
/// @param out out: f32 at out[ids_dst[col] * M + row].
/// @param M weight rows (output width).
/// @param K input width (a multiple of 256).
/// @param ne_get_rows assignment slots (tokens x top-k).
/// @param n_experts experts in the stack (0 for a dense matmul).
/// @param worklist_scratch device scratch for the work list.
/// @param worklist_scratch_bytes bytes of it (ds4_mmq_iq2_xxs_moe_d2r_pair_scratch_bytes).
/// @param stream the stream.
/// @return 0; -1 refused (including no D2R arm on this device, said); -2 / -3 launch failures.
int ds4_mmq_iq2_xxs_moe_d2r_single_launch(const void *W_soa, int64_t soa_blocks, const void *act, const int32_t *ids_dst, const int32_t *expert_bounds, float *out, int M, int K, int64_t ne_get_rows, int n_experts, void *worklist_scratch, size_t worklist_scratch_bytes, cudaStream_t stream);

/** @file
 *  Declaration of ds4_launch_mm_ids_helper, the expert-sorted schedule builder. */
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// ds4 replacement for ggml_cuda_launch_mm_ids_helper (see ds4_mmid.cu).
// Adds the n_expert_used==1 fast path and the large-n global variant that
// upstream asserts against, then delegates to upstream for everything else.
// Always builds the FORWARD id map (upstream's write_inverse=false).
/// Builds the expert-sorted schedule (forward id map only).
/// @param ids the routes, i32 [n_tokens][n_expert_used].
/// @param ids_src1 out: per sorted assignment, its source activation row.
/// @param ids_dst out: per sorted assignment, its output column (token * n_expert_used + slot).
/// @param expert_bounds out: per expert, its first sorted assignment, [n_experts + 1].
/// @param n_experts experts in the stack (0 for a dense matmul).
/// @param n_tokens tokens.
/// @param n_expert_used experts per token.
/// @param nchannels_y activation channels per token (the source row is token * sis1 + slot % nchannels_y).
/// @param si1 ids row stride, elements.
/// @param sis1 activation token stride, rows.
/// @param stream the stream.
void ds4_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1,
        int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used,
        const int nchannels_y, const int si1, const int sis1, cudaStream_t stream);

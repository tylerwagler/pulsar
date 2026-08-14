#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// ds4 replacement for ggml_cuda_launch_mm_ids_helper (see ds4_mmid.cu).
// Adds the n_expert_used==1 fast path and the large-n global variant that
// upstream asserts against, then delegates to upstream for everything else.
// Always builds the FORWARD id map (upstream's write_inverse=false).
void ds4_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1,
        int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used,
        const int nchannels_y, const int si1, const int sis1, cudaStream_t stream);

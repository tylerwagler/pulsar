#pragma once

#include "common.cuh"

// ds4 (L008): upstream declares mul_mat_vec_q_switch_type nowhere -- it is a
// file-local helper in mmvq.cu, reached only through the ggml_tensor dispatch
// entries.  We call it directly (no ggml_tensor available), so mmvq.cu carries
// a one-word patch dropping its `static`, and the declaration lives HERE rather
// than in mmvq.cuh so that upstream header stays byte-identical and re-syncs by
// copy.  Signature must track mmvq.cu:1016.
void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids,
        const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x,
        const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream);

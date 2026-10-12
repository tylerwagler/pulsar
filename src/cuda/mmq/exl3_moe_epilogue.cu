// SPDX-License-Identifier: MIT
// The EXL3 routed MoE's epilogues (exl3_moe_epilogue.cuh has the contracts): the fold and the sum carry the
// Hadamard rotations the format puts on both sides of every projection.  The projections themselves are the
// tensor-core arms' (exl3_moe_prefill.cuh: the decode arm and the prompt-chunk arm), which write UNROTATED z:
// svh and the output Hadamard mix 128 consecutive outputs, and these kernels already touch those elements.
// A decode launch leaves its z as K-split partials; the fold and the sum add the splits in split order
// (task #46).  src/engine/exl3_trellis.h is the format.

#include "exl3_moe_epilogue.cuh"
#include "ds4_exl3_dev.cuh"
#include "cuda/pulsar_cuda_mx.cuh"
#include "engine/exl3_trellis.h"

#include <cstdio>

namespace {

/* the riding shared expert's SwiGLU: warps of 128 elements after the routed fold's */
static int64_t shared_swiglu_tasks(const exl3_moe_shared_swiglu *sh) {
    return sh ? ((int64_t)sh->rows * sh->mid + 127) / 128 : 0;
}
static bool shared_swiglu_ok(const exl3_moe_shared_swiglu *sh) {
    if (sh && (!sh->g || !sh->u || !sh->h || sh->rows <= 0 || sh->mid <= 0)) {
        fprintf(stderr, "ds4_exl3_moe_fold: a shared-expert SwiGLU with a null plane or no rows -- refusing\n");
        return false;
    }
    return true;
}

/* One warp per (pair, 128-block of mid); lane l holds mid columns 4l..4l+3
 * of the block.  The four values per lane make one 32-element MX group an
 * EIGHT-lane group, as in moe_mmq_swiglu_fold_v4_kernel. */
__global__ void __launch_bounds__(256)
exl3_moe_fold_kernel(const float *__restrict__ gate_z,
                     const float *__restrict__ up_z,
                     int z_stride,
                     const int32_t *__restrict__ selected,
                     const float *__restrict__ weights,
                     const void *__restrict__ gate_table,
                     const void *__restrict__ up_table,
                     int up_svh_off,
                     const void *__restrict__ down_table,
                     int in_dim, int mid_dim, int64_t pairs, float clamp,
                     __nv_fp8_e4m3 *__restrict__ mid_q,
                     unsigned char *__restrict__ mid_sf,
                     __nv_bfloat16 *__restrict__ mid_bf16,   /* the format the down arm reads */
                     int mid_kbp,
                     const exl3_moe_shared_swiglu sh,
                     int z_splits, int64_t z_split_stride) {
    const int64_t task = ((int64_t)blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    const int n_blk = mid_dim >> 7;
    const int64_t n_tasks = pairs * n_blk;
    if (task >= n_tasks) {
        /* a riding shared expert's SwiGLU, emitted bf16 for its down (the producer rule): 128 elements of the
         * [rows][mid] planes per warp, the plain silu(g) * u of the family's shared expert */
        const int64_t i0 = (task - n_tasks) * 128 + (int64_t)(threadIdx.x & 31) * 4;
        const int64_t n = (int64_t)sh.rows * sh.mid;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (i0 + j < n) sh.h[i0 + j] = __float2bfloat16(pulsar_swiglu_elem(sh.g[i0 + j], sh.u[i0 + j], 1.0f, 0.0f));
        return;
    }
    const int64_t pair = task / n_blk;
    const int blk = (int)(task - pair * n_blk);
    const int lane = threadIdx.x & 31;
    const int e = selected[pair];
    if (e < 0) return;                                /* another rank's expert (L266 EP) or the router's NaN path */
    const void *const *pg = reinterpret_cast<const void *const *>(gate_table);
    const void *const *pu = reinterpret_cast<const void *const *>(up_table);
    const void *const *pd = reinterpret_cast<const void *const *>(down_table);
    /* gate/up scales: suh[in_dim] | svh[mid_dim]; down scales: suh[mid_dim] | svh[out] */
    const __half *svh_g = reinterpret_cast<const __half *>(pg[2 * (size_t)e + 1]) + in_dim;
    const __half *svh_u = reinterpret_cast<const __half *>(pu[2 * (size_t)e + 1]) + up_svh_off;
    const __half *suh_d = reinterpret_cast<const __half *>(pd[2 * (size_t)e + 1]);
    const int col0 = blk * 128 + lane * 4;
    const float wv = weights[pair];
    const float4 g4 = *reinterpret_cast<const float4 *>(gate_z + pair * z_stride + col0);
    const float4 u4 = *reinterpret_cast<const float4 *>(up_z + pair * z_stride + col0);
    float g[4] = {g4.x, g4.y, g4.z, g4.w}, u[4] = {u4.x, u4.y, u4.z, u4.w};
    for (int s = 1; s < z_splits; ++s) {             /* a decode launch's K splits, in split order */
        const float4 gs = *reinterpret_cast<const float4 *>(gate_z + s * z_split_stride + pair * z_stride + col0);
        const float4 us = *reinterpret_cast<const float4 *>(up_z + s * z_split_stride + pair * z_stride + col0);
        g[0] += gs.x; g[1] += gs.y; g[2] += gs.z; g[3] += gs.w;
        u[0] += us.x; u[1] += us.y; u[2] += us.z; u[3] += us.w;
    }
    exl3dev::had128(g);
    exl3dev::had128(u);
    float t[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float yg = g[j] * __half2float(svh_g[col0 + j]);
        const float yu = u[j] * __half2float(svh_u[col0 + j]);
        t[j] = pulsar_swiglu_elem(yg, yu, wv, clamp) * __half2float(suh_d[col0 + j]);
    }
    exl3dev::had128(t);
    float a = fmaxf(fmaxf(fabsf(t[0]), fabsf(t[1])), fmaxf(fabsf(t[2]), fabsf(t[3])));
#pragma unroll
    for (int off = 4; off > 0; off >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
    const int se = pulsar_mx_shared_exp(a);
    if (mid_q) {
        __nv_fp8_e4m3 *dst = mid_q + (size_t)pair * mid_dim + col0;
        dst[0] = pulsar_mx_encode(t[0], se);
        dst[1] = pulsar_mx_encode(t[1], se);
        dst[2] = pulsar_mx_encode(t[2], se);
        dst[3] = pulsar_mx_encode(t[3], se);
        if ((lane & 7) == 0)
            mid_sf[pulsar_mx_sfoff((int)pair, col0 >> 5, mid_kbp)] = pulsar_mx_scale_byte(se);
    }
    /* L251 / ac69748f: bf16 carries its own exponent, so there is no per-32 block and no scale byte.
     * The fold's output is an op-internal Linear activation, so it follows the family's rule. */
    if (mid_bf16) {
        __nv_bfloat16 *d16 = mid_bf16 + (size_t)pair * mid_dim + col0;
        d16[0] = __float2bfloat16(t[0]);
        d16[1] = __float2bfloat16(t[1]);
        d16[2] = __float2bfloat16(t[2]);
        d16[3] = __float2bfloat16(t[3]);
    }
}

/* Slots whose loads the sum issues together (L284 #7): a decode step's sum is a
 * few warps, latency-bound on its chain of selected -> table -> partial loads
 * when they wait one slot at a time. */
constexpr int kSumBatch = 8;

/* One warp per (token, 128-block of out); slots summed in order, their loads
 * issued kSumBatch slots at a time. */
__global__ void __launch_bounds__(256)
exl3_moe_sum_kernel(float *__restrict__ out,
                    const float *__restrict__ down_z,
                    const int32_t *__restrict__ selected,
                    const void *__restrict__ down_table,
                    int mid_dim, int out_dim, int n_expert, int n_tokens,
                    uint32_t *__restrict__ nf_flag, uint32_t nf_code,
                    const float *__restrict__ shared_y, const float *__restrict__ shared_scale,
                    int z_splits, int64_t z_split_stride) {
    const int64_t task = ((int64_t)blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    const int n_blk = out_dim >> 7;
    if (task >= (int64_t)n_tokens * n_blk) return;
    const int tok = (int)(task / n_blk);
    const int blk = (int)(task - (int64_t)tok * n_blk);
    const int lane = threadIdx.x & 31;
    const int col0 = blk * 128 + lane * 4;
    const void *const *pd = reinterpret_cast<const void *const *>(down_table);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int s0 = 0; s0 < n_expert; s0 += kSumBatch) {
        int ex[kSumBatch];
#pragma unroll
        for (int b = 0; b < kSumBatch; ++b)
            ex[b] = s0 + b < n_expert ? selected[(int64_t)tok * n_expert + s0 + b] : -1;
        float4 z4[kSumBatch];
        __half sv[kSumBatch][4];
#pragma unroll
        for (int b = 0; b < kSumBatch; ++b) {
            if (ex[b] < 0) continue;                  /* past the slots, or another rank's expert (L266 EP): its partial is that rank's */
            const __half *svh_d = reinterpret_cast<const __half *>(pd[2 * (size_t)ex[b] + 1]) + mid_dim;
            z4[b] = *reinterpret_cast<const float4 *>(down_z + ((int64_t)tok * n_expert + s0 + b) * out_dim + col0);
#pragma unroll
            for (int j = 0; j < 4; ++j) sv[b][j] = svh_d[col0 + j];
        }
#pragma unroll
        for (int b = 0; b < kSumBatch; ++b) {
            if (ex[b] < 0) continue;
            float z[4] = {z4[b].x, z4[b].y, z4[b].z, z4[b].w};
            for (int s = 1; s < z_splits; ++s) {     /* a decode launch's K splits, in split order */
                const float4 zs = *reinterpret_cast<const float4 *>(
                    down_z + s * z_split_stride + ((int64_t)tok * n_expert + s0 + b) * out_dim + col0);
                z[0] += zs.x; z[1] += zs.y; z[2] += zs.z; z[3] += zs.w;
            }
            exl3dev::had128(z);
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[j] += z[j] * __half2float(sv[b][j]);
        }
    }
    bool bad = false;
#pragma unroll
    for (int j = 0; j < 4; ++j) bad |= !isfinite(acc[j]);
    if (bad) atomicCAS(nf_flag, 0u, nf_code);
    if (shared_y) {
        /* a riding shared expert's output, added after the routed sum: out + scale * y (scale 1 when absent) */
        const float4 y4 = *reinterpret_cast<const float4 *>(shared_y + (size_t)tok * out_dim + col0);
        const float y[4] = {y4.x, y4.y, y4.z, y4.w};
        const float sc = shared_scale ? shared_scale[tok] : 1.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float s = sc * y[j];
            acc[j] = acc[j] + s;
        }
    }
    *reinterpret_cast<float4 *>(out + (size_t)tok * out_dim + col0) = make_float4(acc[0], acc[1], acc[2], acc[3]);
}

} // namespace

int ds4_exl3_moe_fold_launch(const float *gate_z, const float *up_z, const int32_t *selected,
                             const float *weights, const void *gate_table, const void *up_table,
                             const void *down_table, int in_dim, int mid_dim, int64_t pairs,
                             float clamp, void *mid_q, void *mid_sf, int mid_kbp, cudaStream_t stream, void *mid_bf16,
                             const exl3_moe_shared_swiglu *shared, int z_splits, int64_t z_split_stride) {
    if (!shared_swiglu_ok(shared)) return -1;
    if (!gate_z || !up_z || !selected || !weights || !gate_table || !up_table || !down_table ||
        (!mid_q || !mid_sf) && !mid_bf16 || pairs <= 0 || in_dim <= 0 || mid_dim <= 0 || mid_dim % EXL3_HAD_BLOCK ||
        z_splits < 1) {
        fprintf(stderr, "ds4_exl3_moe_fold_launch: null pointer or bad shape (mid_dim %% 128)\n");
        return -1;
    }
    const int64_t tasks = pairs * (mid_dim >> 7) + shared_swiglu_tasks(shared);
    const unsigned blocks = (unsigned)((tasks + 7) / 8);
    exl3_moe_fold_kernel<<<blocks, 256, 0, stream>>>(
        gate_z, up_z, mid_dim, selected, weights, gate_table, up_table, in_dim, down_table, in_dim, mid_dim, pairs,
        clamp, (__nv_fp8_e4m3 *)mid_q, (unsigned char *)mid_sf, (__nv_bfloat16 *)mid_bf16, mid_kbp,
        shared ? *shared : exl3_moe_shared_swiglu{}, z_splits, z_split_stride);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4_exl3_moe_fold_launch: launch failed: %s\n", cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

int ds4_exl3_moe_fold_fused_launch(const float *gate_up_z, const int32_t *selected, const float *weights,
                                   const void *gate_up_table, const void *down_table, int in_dim, int mid_dim,
                                   int64_t pairs, float clamp, void *mid_q, void *mid_sf, int mid_kbp,
                                   cudaStream_t stream, void *mid_bf16, const exl3_moe_shared_swiglu *shared,
                                   int z_splits, int64_t z_split_stride) {
    if (!shared_swiglu_ok(shared)) return -1;
    if (!gate_up_z || !selected || !weights || !gate_up_table || !down_table || (!mid_q || !mid_sf) && !mid_bf16 || pairs <= 0 ||
        in_dim <= 0 || mid_dim <= 0 || mid_dim % EXL3_HAD_BLOCK || z_splits < 1) {
        fprintf(stderr, "ds4_exl3_moe_fold_fused_launch: null pointer or bad shape (mid_dim %% 128)\n");
        return -1;
    }
    /* gate = rows 0..mid-1 of the fused output, up = rows mid..2 mid-1; the scales
     * plane is suh [in_dim] | svh [2 mid], so up's svh starts at in_dim + mid */
    const int64_t tasks = pairs * (mid_dim >> 7) + shared_swiglu_tasks(shared);
    const unsigned blocks = (unsigned)((tasks + 7) / 8);
    exl3_moe_fold_kernel<<<blocks, 256, 0, stream>>>(
        gate_up_z, gate_up_z + mid_dim, 2 * mid_dim, selected, weights, gate_up_table, gate_up_table, in_dim + mid_dim,
        down_table, in_dim, mid_dim, pairs, clamp, (__nv_fp8_e4m3 *)mid_q, (unsigned char *)mid_sf, (__nv_bfloat16 *)mid_bf16, mid_kbp,
        shared ? *shared : exl3_moe_shared_swiglu{}, z_splits, z_split_stride);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4_exl3_moe_fold_fused_launch: launch failed: %s\n", cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

int ds4_exl3_moe_sum_launch(float *out, const float *down_z, const int32_t *selected,
                            const void *down_table, int mid_dim, int out_dim, int n_expert,
                            int n_tokens, uint32_t *nf_flag, uint32_t nf_code, cudaStream_t stream,
                            const float *shared_y, const float *shared_scale, int z_splits, int64_t z_split_stride) {
    if (!out || !down_z || !selected || !down_table || !nf_flag || mid_dim <= 0 || out_dim <= 0 ||
        out_dim % EXL3_HAD_BLOCK || n_expert <= 0 || n_tokens <= 0 || z_splits < 1) {
        fprintf(stderr, "ds4_exl3_moe_sum_launch: null pointer or bad shape (out_dim %% 128)\n");
        return -1;
    }
    const int64_t tasks = (int64_t)n_tokens * (out_dim >> 7);
    const unsigned blocks = (unsigned)((tasks + 7) / 8);
    exl3_moe_sum_kernel<<<blocks, 256, 0, stream>>>(
        out, down_z, selected, down_table, mid_dim, out_dim, n_expert, n_tokens, nf_flag, nf_code, shared_y, shared_scale,
        z_splits, z_split_stride);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4_exl3_moe_sum_launch: launch failed: %s\n", cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

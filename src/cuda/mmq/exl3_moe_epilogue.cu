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

#include <cstdint>
#include <cstdio>

namespace {

/* One warp per (pair, 128-block of mid); lane l holds mid columns 4l..4l+3
 * of the block.  The four values per lane make one 32-element MX group an
 * EIGHT-lane group, as in moe_mmq_swiglu_fold_v4_kernel.  UP_ONLY (L296): the
 * up-only fold -- no gate projection (gate_z / gate_table unread), the
 * activation relu(y_u)^2 (pulsar_relu2_elem) with the route weight folded in;
 * every other step is the gated fold's, unchanged. */
template <bool UP_ONLY>
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
                     int z_splits, int64_t z_split_stride) {
    const int64_t task = ((int64_t)blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    const int n_blk = mid_dim >> 7;
    if (task >= pairs * n_blk) return;
    const int64_t pair = task / n_blk;
    const int blk = (int)(task - pair * n_blk);
    const int lane = threadIdx.x & 31;
    const int e = selected[pair];
    if (e < 0) return;                                /* another rank's expert (L266 EP) or the router's NaN path */
    const void *const *pu = reinterpret_cast<const void *const *>(up_table);
    const void *const *pd = reinterpret_cast<const void *const *>(down_table);
    /* gate/up scales: suh[in_dim] | svh[mid_dim]; down scales: suh[mid_dim] | svh[out] */
    const __half *svh_u = reinterpret_cast<const __half *>(pu[2 * (size_t)e + 1]) + up_svh_off;
    const __half *suh_d = reinterpret_cast<const __half *>(pd[2 * (size_t)e + 1]);
    const int col0 = blk * 128 + lane * 4;
    const float wv = weights[pair];
    float g[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if constexpr (!UP_ONLY) {
        const float4 g4 = *reinterpret_cast<const float4 *>(gate_z + pair * z_stride + col0);
        g[0] = g4.x; g[1] = g4.y; g[2] = g4.z; g[3] = g4.w;
    }
    const float4 u4 = *reinterpret_cast<const float4 *>(up_z + pair * z_stride + col0);
    float u[4] = {u4.x, u4.y, u4.z, u4.w};
    for (int s = 1; s < z_splits; ++s) {             /* a decode launch's K splits, in split order */
        if constexpr (!UP_ONLY) {
            const float4 gs = *reinterpret_cast<const float4 *>(gate_z + s * z_split_stride + pair * z_stride + col0);
            g[0] += gs.x; g[1] += gs.y; g[2] += gs.z; g[3] += gs.w;
        }
        const float4 us = *reinterpret_cast<const float4 *>(up_z + s * z_split_stride + pair * z_stride + col0);
        u[0] += us.x; u[1] += us.y; u[2] += us.z; u[3] += us.w;
    }
    if constexpr (!UP_ONLY) exl3dev::had128(g);
    exl3dev::had128(u);
    float t[4];
    if constexpr (UP_ONLY) {
        (void)gate_table; (void)clamp; (void)in_dim;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float yu = u[j] * __half2float(svh_u[col0 + j]);
            t[j] = pulsar_relu2_elem(yu, wv) * __half2float(suh_d[col0 + j]);
        }
    } else {
        const void *const *pg = reinterpret_cast<const void *const *>(gate_table);
        const __half *svh_g = reinterpret_cast<const __half *>(pg[2 * (size_t)e + 1]) + in_dim;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float yg = g[j] * __half2float(svh_g[col0 + j]);
            const float yu = u[j] * __half2float(svh_u[col0 + j]);
            t[j] = pulsar_swiglu_elem(yg, yu, wv, clamp) * __half2float(suh_d[col0 + j]);
        }
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

/* The sum's CTA: up to kSumWarpsMax warps (one per slot, at most), the slots staged up to kSumChunk at a time in
 * dynamic shared memory sized by sum_chunk(n_expert) (a 6-slot router's CTA keeps 6 KiB, not 32). */
constexpr int kSumWarpsMax = 16;
constexpr int kSumChunk = 32;
__host__ __device__ constexpr int sum_chunk(int n_expert) { return n_expert < kSumChunk ? n_expert : kSumChunk; }
constexpr size_t sum_smem_bytes(int chunk) { return (size_t)chunk * 32 * 2 * sizeof(float4) + (size_t)chunk * sizeof(int); }

/* One CTA per (token, 128-block of out).  Each slot's chain -- selected -> table -> svh, its split partials added in
 * split order, the Hadamard -- runs on its own warp (warp w takes slots w, w + warps, ...), which leaves the rotated
 * z and svh in shared memory; warp 0 then sums the slots in slot order, the same fma per element in the same order
 * as one warp walking the slots (L296 dec2: that walk, a few warps each loading slot after slot, was 20 us per
 * decode-step layer). */
__global__ void __launch_bounds__(kSumWarpsMax * 32)
exl3_moe_sum_kernel(float *__restrict__ out,
                    const float *__restrict__ down_z,
                    const int32_t *__restrict__ selected,
                    const void *__restrict__ down_table,
                    int mid_dim, int out_dim, int n_expert,
                    uint32_t *__restrict__ nf_flag, uint32_t nf_code,
                    const exl3_moe_shared_add shared, int z_splits, int64_t z_split_stride) {
    extern __shared__ float4 s_sum[];
    const int chunk = sum_chunk(n_expert);
    float4 *s_z = s_sum, *s_sv = s_sum + chunk * 32;   /* [chunk][32] each, then s_live [chunk] */
    int *s_live = reinterpret_cast<int *>(s_sum + 2 * chunk * 32);
    const int n_blk = out_dim >> 7;
    const int tok = (int)(blockIdx.x / (unsigned)n_blk);
    const int blk = (int)(blockIdx.x - (unsigned)tok * (unsigned)n_blk);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
    const int col0 = blk * 128 + lane * 4;
    const void *const *pd = reinterpret_cast<const void *const *>(down_table);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int c0 = 0; c0 < n_expert; c0 += chunk) {
        const int nc = min(chunk, n_expert - c0);
        for (int b = warp; b < nc; b += warps) {
            const int64_t pair = (int64_t)tok * n_expert + c0 + b;
            const int ex = selected[pair];
            if (ex < 0) {                             /* another rank's expert (L266 EP): its partial is that rank's */
                if (lane == 0) s_live[b] = 0;
                continue;
            }
            const __half *svh_d = reinterpret_cast<const __half *>(pd[2 * (size_t)ex + 1]) + mid_dim;
            const float4 z4 = *reinterpret_cast<const float4 *>(down_z + pair * out_dim + col0);
            float z[4] = {z4.x, z4.y, z4.z, z4.w};
            for (int s = 1; s < z_splits; ++s) {     /* a decode launch's K splits, in split order */
                const float4 zs = *reinterpret_cast<const float4 *>(down_z + s * z_split_stride + pair * out_dim + col0);
                z[0] += zs.x; z[1] += zs.y; z[2] += zs.z; z[3] += zs.w;
            }
            exl3dev::had128(z);
            s_z[b * 32 + lane] = make_float4(z[0], z[1], z[2], z[3]);
            s_sv[b * 32 + lane] = make_float4(__half2float(svh_d[col0]), __half2float(svh_d[col0 + 1]),
                                        __half2float(svh_d[col0 + 2]), __half2float(svh_d[col0 + 3]));
            if (lane == 0) s_live[b] = 1;
        }
        __syncthreads();
        if (warp == 0)
            for (int b = 0; b < nc; ++b) {
                if (!s_live[b]) continue;
                const float4 z4 = s_z[b * 32 + lane], v4 = s_sv[b * 32 + lane];
                acc[0] += z4.x * v4.x;
                acc[1] += z4.y * v4.y;
                acc[2] += z4.z * v4.z;
                acc[3] += z4.w * v4.w;
            }
        __syncthreads();
    }
    if (warp != 0) return;
    bool bad = false;
#pragma unroll
    for (int j = 0; j < 4; ++j) bad |= !isfinite(acc[j]);
    if (bad) atomicCAS(nf_flag, 0u, nf_code);
#pragma unroll
    for (int e = 0; e < PULSAR_ROUTER_SHARED_MAX; ++e) {
        if (e >= shared.n) break;
        /* the riding shared experts' outputs, in order, after the routed sum: out + scale * y (scale 1 when absent) */
        const float4 y4 = *reinterpret_cast<const float4 *>(shared.y[e] + (size_t)tok * out_dim + col0);
        const float y[4] = {y4.x, y4.y, y4.z, y4.w};
        const float sc = shared.scale[e] ? shared.scale[e][tok] : 1.0f;
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
                             int z_splits, int64_t z_split_stride) {
    if (!gate_z || !up_z || !selected || !weights || !gate_table || !up_table || !down_table ||
        (!mid_q || !mid_sf) && !mid_bf16 || pairs <= 0 || in_dim <= 0 || mid_dim <= 0 || mid_dim % EXL3_HAD_BLOCK ||
        z_splits < 1) {
        fprintf(stderr, "ds4_exl3_moe_fold_launch: null pointer or bad shape (mid_dim %% 128)\n");
        return -1;
    }
    const int64_t tasks = pairs * (mid_dim >> 7);
    const unsigned blocks = (unsigned)((tasks + 7) / 8);
    exl3_moe_fold_kernel<false><<<blocks, 256, 0, stream>>>(
        gate_z, up_z, mid_dim, selected, weights, gate_table, up_table, in_dim, down_table, in_dim, mid_dim, pairs,
        clamp, (__nv_fp8_e4m3 *)mid_q, (unsigned char *)mid_sf, (__nv_bfloat16 *)mid_bf16, mid_kbp, z_splits,
        z_split_stride);
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
                                   cudaStream_t stream, void *mid_bf16, int z_splits, int64_t z_split_stride) {
    if (!gate_up_z || !selected || !weights || !gate_up_table || !down_table || (!mid_q || !mid_sf) && !mid_bf16 || pairs <= 0 ||
        in_dim <= 0 || mid_dim <= 0 || mid_dim % EXL3_HAD_BLOCK || z_splits < 1) {
        fprintf(stderr, "ds4_exl3_moe_fold_fused_launch: null pointer or bad shape (mid_dim %% 128)\n");
        return -1;
    }
    /* gate = rows 0..mid-1 of the fused output, up = rows mid..2 mid-1; the scales
     * plane is suh [in_dim] | svh [2 mid], so up's svh starts at in_dim + mid */
    const int64_t tasks = pairs * (mid_dim >> 7);
    const unsigned blocks = (unsigned)((tasks + 7) / 8);
    exl3_moe_fold_kernel<false><<<blocks, 256, 0, stream>>>(
        gate_up_z, gate_up_z + mid_dim, 2 * mid_dim, selected, weights, gate_up_table, gate_up_table, in_dim + mid_dim,
        down_table, in_dim, mid_dim, pairs, clamp, (__nv_fp8_e4m3 *)mid_q, (unsigned char *)mid_sf, (__nv_bfloat16 *)mid_bf16, mid_kbp,
        z_splits, z_split_stride);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4_exl3_moe_fold_fused_launch: launch failed: %s\n", cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

int ds4_exl3_moe_fold_up_launch(const float *up_z, const int32_t *selected, const float *weights, const void *up_table,
                                const void *down_table, int in_dim, int mid_dim, int64_t pairs, void *mid_q, void *mid_sf,
                                int mid_kbp, cudaStream_t stream, void *mid_bf16, int z_splits, int64_t z_split_stride) {
    if (!up_z || !selected || !weights || !up_table || !down_table || (!mid_q || !mid_sf) && !mid_bf16 || pairs <= 0 ||
        in_dim <= 0 || mid_dim <= 0 || mid_dim % EXL3_HAD_BLOCK || z_splits < 1) {
        fprintf(stderr, "ds4_exl3_moe_fold_up_launch: null pointer or bad shape (mid_dim %% 128)\n");
        return -1;
    }
    const int64_t tasks = pairs * (mid_dim >> 7);
    const unsigned blocks = (unsigned)((tasks + 7) / 8);
    exl3_moe_fold_kernel<true><<<blocks, 256, 0, stream>>>(
        nullptr, up_z, mid_dim, selected, weights, nullptr, up_table, in_dim, down_table, in_dim, mid_dim, pairs, 0.0f,
        (__nv_fp8_e4m3 *)mid_q, (unsigned char *)mid_sf, (__nv_bfloat16 *)mid_bf16, mid_kbp, z_splits, z_split_stride);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4_exl3_moe_fold_up_launch: launch failed: %s\n", cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

int ds4_exl3_moe_sum_launch(float *out, const float *down_z, const int32_t *selected,
                            const void *down_table, int mid_dim, int out_dim, int n_expert,
                            int n_tokens, uint32_t *nf_flag, uint32_t nf_code, cudaStream_t stream,
                            const exl3_moe_shared_add *shared, int z_splits, int64_t z_split_stride) {
    if (!out || !down_z || !selected || !down_table || !nf_flag || mid_dim <= 0 || out_dim <= 0 ||
        out_dim % EXL3_HAD_BLOCK || n_expert <= 0 || n_tokens <= 0 || z_splits < 1 ||
        (shared && (shared->n < 0 || shared->n > PULSAR_ROUTER_SHARED_MAX))) {
        fprintf(stderr, "ds4_exl3_moe_sum_launch: null pointer or bad shape (out_dim %% 128, 0..%d shared)\n",
                PULSAR_ROUTER_SHARED_MAX);
        return -1;
    }
    exl3_moe_shared_add sh{};
    if (shared) sh = *shared;
    for (int e = 0; e < sh.n; ++e)
        if (!sh.y[e]) {
            fprintf(stderr, "ds4_exl3_moe_sum_launch: shared expert %d has no output -- refusing\n", e);
            return -1;
        }
    const int64_t blocks = (int64_t)n_tokens * (out_dim >> 7);
    if (blocks > INT32_MAX) {
        fprintf(stderr, "ds4_exl3_moe_sum_launch: %d tokens x %d outputs is past one launch's grid -- refusing\n",
                n_tokens, out_dim);
        return -1;
    }
    const int warps = n_expert < kSumWarpsMax ? n_expert : kSumWarpsMax;
    exl3_moe_sum_kernel<<<(unsigned)blocks, warps * 32, sum_smem_bytes(sum_chunk(n_expert)), stream>>>(
        out, down_z, selected, down_table, mid_dim, out_dim, n_expert, nf_flag, nf_code, sh, z_splits,
        z_split_stride);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4_exl3_moe_sum_launch: launch failed: %s\n", cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

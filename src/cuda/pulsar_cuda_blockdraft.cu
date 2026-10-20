/* The block drafter's small ops (L298).  The contract is pulsar_cuda_blockdraft.h. */
#include "pulsar_cuda_blockdraft.h"

#include <cuda_bf16.h>

#include <math.h>
#include <stdio.h>

namespace {

__device__ __forceinline__ float bf(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ __forceinline__ uint16_t to_bf(float x) {
    const __nv_bfloat16 h = __float2bfloat16_rn(x);
    return *reinterpret_cast<const uint16_t *>(&h);
}

int refuse(const char *what) {
    fprintf(stderr, "pulsar: block drafter op refused: %s\n", what);
    return -1;
}
int launched(const char *what) {
    const cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) return 0;
    fprintf(stderr, "pulsar: block drafter %s launch failed: %s\n", what, cudaGetErrorString(e));
    return -3;
}

/* one thread per (row, channel) */
__global__ void conv_kernel(const uint16_t *__restrict__ x, uint64_t ld_x, const uint16_t *__restrict__ delta,
                            uint64_t ld_delta, const uint16_t *__restrict__ base, uint32_t taps, uint32_t group,
                            uint32_t block, uint32_t width, uint16_t *__restrict__ out, uint64_t ld_out) {
    const uint32_t r = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= width) return;
    const uint32_t groups = width / group, g = c / group, p = r % block;
    const uint16_t *d = delta + (uint64_t)r * ld_delta;
    float acc = (bf(base[c]) + bf(d[g])) * bf(x[(uint64_t)r * ld_x + c]);
    for (uint32_t t = 1; t < taps && t <= p; t++)
        acc = fmaf(bf(base[(uint64_t)t * width + c]) + bf(d[t * groups + g]), bf(x[(uint64_t)(r - t) * ld_x + c]), acc);
    out[(uint64_t)r * ld_out + c] = to_bf(acc);
}

__global__ void swiglu_kernel(const uint16_t *__restrict__ g, uint64_t ld_g, const uint16_t *__restrict__ u,
                              uint64_t ld_u, uint32_t width, uint16_t *__restrict__ h, uint64_t ld_h) {
    const uint32_t r = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= width) return;
    const float a = bf(g[(uint64_t)r * ld_g + c]);
    const float s = a / (1.0f + expf(-a));
    h[(uint64_t)r * ld_h + c] = to_bf(s * bf(u[(uint64_t)r * ld_u + c]));
}

__global__ void rows_copy_kernel(const uint16_t *__restrict__ src, uint64_t ld_src, uint32_t width8,
                                 uint16_t *__restrict__ dst, uint64_t ld_dst) {
    const uint32_t r = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width8) return;
    reinterpret_cast<uint4 *>(dst + (uint64_t)r * ld_dst)[i] = reinterpret_cast<const uint4 *>(src + (uint64_t)r * ld_src)[i];
}

/* one thread per 8 columns of a row: the streams summed in order, then the mean */
__global__ void stream_mean_kernel(const uint16_t *__restrict__ src, uint64_t ld_src, uint32_t n_streams,
                                   uint32_t width8, uint32_t width, uint16_t *__restrict__ dst, uint64_t ld_dst) {
    const uint32_t r = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width8) return;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (uint32_t st = 0; st < n_streams; st++) {
        const uint4 v = reinterpret_cast<const uint4 *>(src + (uint64_t)r * ld_src + (uint64_t)st * width)[i];
        const uint32_t w[4] = {v.x, v.y, v.z, v.w};
        for (int j = 0; j < 4; j++) {
            acc[2 * j] += bf((uint16_t)(w[j] & 0xffffu));
            acc[2 * j + 1] += bf((uint16_t)(w[j] >> 16));
        }
    }
    const float inv = __frcp_rn((float)n_streams);   /* IEEE reciprocal (the build is fast-math): exact for 2, 4, 8 */
    uint32_t o[4];
    for (int j = 0; j < 4; j++) o[j] = (uint32_t)to_bf(acc[2 * j] * inv) | (uint32_t)to_bf(acc[2 * j + 1] * inv) << 16;
    reinterpret_cast<uint4 *>(dst + (uint64_t)r * ld_dst)[i] = make_uint4(o[0], o[1], o[2], o[3]);
}

constexpr int TOPK_THREADS = 256;

/* a > b in the selection order: the larger value, ties to the lower id */
__device__ __forceinline__ bool before(float va, int32_t ia, float vb, int32_t ib) {
    return va > vb || (va == vb && ia < ib);
}

/* one CTA per row: each thread a sorted local top-k of its strided columns and an online (max, sum exp); the CTA's
 * logsumexp, then k rounds of a CTA argmax over the threads' list heads */
template <int KM>
__global__ void __launch_bounds__(TOPK_THREADS) topk_kernel(const uint16_t *__restrict__ logits, uint64_t ld,
                                                           uint32_t n, const int32_t *__restrict__ id_map,
                                                           int32_t *ids, float *lp) {
    constexpr uint32_t k = (uint32_t)KM;
    const uint32_t row = blockIdx.x, tid = threadIdx.x;
    const uint16_t *x = logits + (uint64_t)row * ld;
    /* the thread's sorted list in REGISTERS: k is the template's and every index below a compile-time constant
     * (fully unrolled), so the list never lives in local memory (L299) */
    float lv[KM];
    int32_t li[KM];
#pragma unroll
    for (int j = 0; j < KM; j++) { lv[j] = -INFINITY; li[j] = INT32_MAX; }
    const int kk = (int)k;
    float m = -INFINITY, s = 0.f;
    for (uint32_t c = tid; c < n; c += TOPK_THREADS) {
        const float v = bf(x[c]);
        if (v > m) { s = s * expf(m - v) + 1.f; m = v; }
        else s += expf(v - m);
        float lk = -INFINITY;
        int32_t lik = INT32_MAX;
#pragma unroll
        for (int j = 0; j < KM; j++)
            if (j == kk - 1) { lk = lv[j]; lik = li[j]; }
        if (!before(v, (int32_t)c, lk, lik)) continue;
        /* the insertion: from the list's end, shift each entry v outranks down one; v takes the first it does not */
        bool placed = false;
#pragma unroll
        for (int j = KM - 1; j >= 0; j--) {
            if (placed || j > kk - 1) continue;
            if (j > 0 && before(v, (int32_t)c, lv[j - 1], li[j - 1])) { lv[j] = lv[j - 1]; li[j] = li[j - 1]; }
            else { lv[j] = v; li[j] = (int32_t)c; placed = true; }
        }
    }
    __shared__ float sm[TOPK_THREADS], ss[TOPK_THREADS];
    __shared__ int32_t si[TOPK_THREADS];
    sm[tid] = m;
    ss[tid] = s;
    __syncthreads();
    for (uint32_t w = TOPK_THREADS / 2; w > 0; w >>= 1) {
        if (tid < w) {
            const float m1 = sm[tid], m2 = sm[tid + w], mn = fmaxf(m1, m2);
            const float s1 = m1 == -INFINITY ? 0.f : ss[tid] * expf(m1 - mn);
            const float s2 = m2 == -INFINITY ? 0.f : ss[tid + w] * expf(m2 - mn);
            sm[tid] = mn;
            ss[tid] = s1 + s2;
        }
        __syncthreads();
    }
    const float lse = sm[0] + logf(ss[0]);
    __syncthreads();
    int head = 0;
    for (uint32_t j = 0; j < k; j++) {
        float hv = -INFINITY;
        int32_t hi = INT32_MAX;
#pragma unroll
        for (int q = 0; q < KM; q++)
            if (q == head && q < kk) { hv = lv[q]; hi = li[q]; }
        sm[tid] = hv;
        si[tid] = hi;
        ss[tid] = (float)tid;
        __syncthreads();
        for (uint32_t w = TOPK_THREADS / 2; w > 0; w >>= 1) {
            if (tid < w && before(sm[tid + w], si[tid + w], sm[tid], si[tid])) {
                sm[tid] = sm[tid + w];
                si[tid] = si[tid + w];
                ss[tid] = ss[tid + w];
            }
            __syncthreads();
        }
        if (tid == 0) {
            ids[(uint64_t)row * k + j] = id_map ? id_map[si[0]] : si[0];
            lp[(uint64_t)row * k + j] = sm[0] - lse;
        }
        if ((float)tid == ss[0]) head++;
        __syncthreads();
    }
}

}  // namespace

int pulsar_blockdraft_conv_launch(const uint16_t *x, uint64_t ld_x, const uint16_t *delta, uint64_t ld_delta,
                                  const uint16_t *base, uint32_t taps, uint32_t group, uint32_t block, uint32_t n_rows,
                                  uint32_t width, uint16_t *out, uint64_t ld_out, cudaStream_t stream) {
    if (!x || !delta || !base || !out || taps == 0 || group == 0 || width % group || block == 0 || n_rows % block ||
        n_rows == 0 || n_rows > 65535u || ld_x < width || ld_out < width ||
        ld_delta < (uint64_t)taps * (width / group))
        return refuse("grouped conv: a NULL buffer, rows not whole blocks, or a group / stride that does not tile");
    conv_kernel<<<dim3((width + 255u) / 256u, n_rows), 256, 0, stream>>>(x, ld_x, delta, ld_delta, base, taps, group,
                                                                         block, width, out, ld_out);
    return launched("grouped conv");
}

int pulsar_blockdraft_swiglu_launch(const uint16_t *g, uint64_t ld_g, const uint16_t *u, uint64_t ld_u,
                                    uint32_t n_rows, uint32_t width, uint16_t *h, uint64_t ld_h, cudaStream_t stream) {
    if (!g || !u || !h || n_rows == 0 || n_rows > 65535u || width == 0)
        return refuse("swiglu: a NULL buffer or no rows");
    swiglu_kernel<<<dim3((width + 255u) / 256u, n_rows), 256, 0, stream>>>(g, ld_g, u, ld_u, width, h, ld_h);
    return launched("swiglu");
}

int pulsar_blockdraft_rows_copy_launch(const uint16_t *src, uint64_t ld_src, uint32_t n_rows, uint32_t width,
                                       uint16_t *dst, uint64_t ld_dst, cudaStream_t stream) {
    if (!src || !dst || n_rows == 0 || n_rows > 65535u || width % 8u || ld_src % 8u || ld_dst % 8u ||
        ((uintptr_t)src & 15u) || ((uintptr_t)dst & 15u))
        return refuse("row copy: rows not whole 16-byte units");
    const uint32_t w8 = width / 8u;
    rows_copy_kernel<<<dim3((w8 + 127u) / 128u, n_rows), 128, 0, stream>>>(src, ld_src, w8, dst, ld_dst);
    return launched("row copy");
}

int pulsar_blockdraft_stream_mean_launch(const uint16_t *src, uint64_t ld_src, uint32_t n_streams, uint32_t n_rows,
                                         uint32_t width, uint16_t *dst, uint64_t ld_dst, cudaStream_t stream) {
    if (!src || !dst || n_streams == 0 || n_rows == 0 || n_rows > 65535u || width % 8u || ld_src % 8u || ld_dst % 8u ||
        ld_src < (uint64_t)n_streams * width || ((uintptr_t)src & 15u) || ((uintptr_t)dst & 15u))
        return refuse("stream mean: no streams or rows, or rows not whole 16-byte units");
    const uint32_t w8 = width / 8u;
    stream_mean_kernel<<<dim3((w8 + 127u) / 128u, n_rows), 128, 0, stream>>>(src, ld_src, n_streams, w8, width, dst,
                                                                             ld_dst);
    return launched("stream mean");
}

int pulsar_rows_topk_logprob_launch(const uint16_t *logits, uint64_t ld, uint32_t n_rows, uint32_t n, uint32_t k,
                                    const int32_t *id_map, int32_t *ids, float *lp, cudaStream_t stream) {
    if (!logits || !ids || !lp || n_rows == 0 || n == 0 || ld < n || k == 0 || k > PULSAR_BLOCKDRAFT_TOPK_MAX || k > n)
        return refuse("top-k: a NULL buffer, no rows, or k outside 1 .. 16");
    static_assert(PULSAR_BLOCKDRAFT_TOPK_MAX == 16, "one instantiation a k, 1 .. 16");
    switch (k) {   /* one instantiation a k: the list's length is a compile-time constant */
#define PULSAR_TOPK_CASE(K) case K: topk_kernel<K><<<n_rows, TOPK_THREADS, 0, stream>>>(logits, ld, n, id_map, ids, lp); break;
        PULSAR_TOPK_CASE(1) PULSAR_TOPK_CASE(2) PULSAR_TOPK_CASE(3) PULSAR_TOPK_CASE(4) PULSAR_TOPK_CASE(5)
        PULSAR_TOPK_CASE(6) PULSAR_TOPK_CASE(7) PULSAR_TOPK_CASE(8) PULSAR_TOPK_CASE(9) PULSAR_TOPK_CASE(10)
        PULSAR_TOPK_CASE(11) PULSAR_TOPK_CASE(12) PULSAR_TOPK_CASE(13) PULSAR_TOPK_CASE(14) PULSAR_TOPK_CASE(15)
        PULSAR_TOPK_CASE(16)
#undef PULSAR_TOPK_CASE
    }
    return launched("top-k");
}

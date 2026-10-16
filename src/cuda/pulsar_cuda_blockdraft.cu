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

constexpr int TOPK_THREADS = 256;

/* a > b in the selection order: the larger value, ties to the lower id */
__device__ __forceinline__ bool before(float va, int32_t ia, float vb, int32_t ib) {
    return va > vb || (va == vb && ia < ib);
}

/* one CTA per row: each thread a sorted local top-k of its strided columns and an online (max, sum exp); the CTA's
 * logsumexp, then k rounds of a CTA argmax over the threads' list heads */
__global__ void __launch_bounds__(TOPK_THREADS) topk_kernel(const uint16_t *__restrict__ logits, uint64_t ld,
                                                           uint32_t n, uint32_t k, int32_t *ids, float *lp) {
    const uint32_t row = blockIdx.x, tid = threadIdx.x;
    const uint16_t *x = logits + (uint64_t)row * ld;
    float lv[PULSAR_BLOCKDRAFT_TOPK_MAX];
    int32_t li[PULSAR_BLOCKDRAFT_TOPK_MAX];
    for (uint32_t j = 0; j < k; j++) { lv[j] = -INFINITY; li[j] = INT32_MAX; }
    float m = -INFINITY, s = 0.f;
    for (uint32_t c = tid; c < n; c += TOPK_THREADS) {
        const float v = bf(x[c]);
        if (v > m) { s = s * expf(m - v) + 1.f; m = v; }
        else s += expf(v - m);
        if (!before(v, (int32_t)c, lv[k - 1], li[k - 1])) continue;
        uint32_t j = k - 1;
        while (j > 0 && before(v, (int32_t)c, lv[j - 1], li[j - 1])) { lv[j] = lv[j - 1]; li[j] = li[j - 1]; j--; }
        lv[j] = v;
        li[j] = (int32_t)c;
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
    uint32_t head = 0;
    for (uint32_t j = 0; j < k; j++) {
        sm[tid] = head < k ? lv[head] : -INFINITY;
        si[tid] = head < k ? li[head] : INT32_MAX;
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
            ids[(uint64_t)row * k + j] = si[0];
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

int pulsar_rows_topk_logprob_launch(const uint16_t *logits, uint64_t ld, uint32_t n_rows, uint32_t n, uint32_t k,
                                    int32_t *ids, float *lp, cudaStream_t stream) {
    if (!logits || !ids || !lp || n_rows == 0 || n == 0 || ld < n || k == 0 || k > PULSAR_BLOCKDRAFT_TOPK_MAX || k > n)
        return refuse("top-k: a NULL buffer, no rows, or k outside 1 .. 16");
    topk_kernel<<<n_rows, TOPK_THREADS, 0, stream>>>(logits, ld, n, k, ids, lp);
    return launched("top-k");
}

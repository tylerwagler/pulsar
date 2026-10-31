/** @file
 *  Model-head family options: the embedding gather + norm and the logits finish (scale, padding mask). */
/* The model head's family options (L295 E5): the embedding norm and the logits
 * finish (scale + padding mask).  Contracts in pulsar_cuda_head.h. */
#include "pulsar_cuda_head.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

namespace {

/// Block size of head_embed_kernel (8 warps; one CTA a token row).
constexpr int kEmbedThreads = 256;
/// Block size of head_finish_kernel.
constexpr int kFinishThreads = 256;

/// bf16 bits -> f32 (exact widening).
/// @param b bf16 bits
/// @return the value as f32
__device__ __forceinline__ float bf2f(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
/// f32 -> bf16 bits, round to nearest even.
/// @param f the value
/// @return its bf16 bits
__device__ __forceinline__ uint16_t f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }

/* One CTA a row.  Thread i owns the 8-element chunks i, i + 256, ...; its squares are an f32 fma chain over its
 * chunks in order, the warp folds by xor-shuffle, warp 0 adds the eight warp sums in order. */
/** Embedding gather + optional RMS norm.  Grid: n token rows (blockIdx.x = row t); block kEmbedThreads; no
 *  dynamic smem.  out[t] = table[ids[t]] * rsqrt(mean(x^2) + eps) * norm_w (no 1 + w), or the row copied as is
 *  when norm_w is NULL; an id outside [0, n_vocab) writes a bf16 NaN row.
 *  @param table   device bf16 [n_vocab][n_embd] viewed as uint4 (8 bf16 a chunk), 16-byte aligned
 *  @param n_vocab rows in `table`
 *  @param ids     device [n] token ids
 *  @param n_embd  row width in elements (a multiple of 8)
 *  @param norm_w  device bf16 [n_embd] norm weight as uint4, or NULL: no norm
 *  @param eps     the norm's epsilon (added to the mean square)
 *  @param out     device bf16 output rows, row t at out + t * ld_out (16-byte aligned)
 *  @param ld_out  output row pitch in elements (a multiple of 8) */
__global__ void __launch_bounds__(kEmbedThreads)
head_embed_kernel(const uint4 *__restrict__ table, uint32_t n_vocab, const int32_t *__restrict__ ids, uint32_t n_embd,
                  const uint4 *__restrict__ norm_w, float eps, uint16_t *__restrict__ out, uint64_t ld_out) {
    __shared__ float warp_ss[kEmbedThreads / 32];
    __shared__ float s_rstd;
    const uint32_t t = blockIdx.x, chunks = n_embd / 8;
    uint4 *dst = (uint4 *)(out + (uint64_t)t * ld_out);
    const int32_t id = ids[t];
    if (id < 0 || (uint32_t)id >= n_vocab) {   /* the engine checks ids first; a stray one is loud, never another row */
        const uint4 nan = make_uint4(0x7fc07fc0u, 0x7fc07fc0u, 0x7fc07fc0u, 0x7fc07fc0u);
        for (uint32_t i = threadIdx.x; i < chunks; i += kEmbedThreads) dst[i] = nan;
        return;
    }
    const uint4 *src = table + (uint64_t)id * chunks;
    if (!norm_w) {
        for (uint32_t i = threadIdx.x; i < chunks; i += kEmbedThreads) dst[i] = src[i];
        return;
    }
    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < chunks; i += kEmbedThreads) {
        const uint4 v = src[i];
        const uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float a = bf2f((uint16_t)(u[k] & 0xffffu)), b = bf2f((uint16_t)(u[k] >> 16));
            ss = __fmaf_rn(a, a, ss);
            ss = __fmaf_rn(b, b, ss);
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, o));
    if ((threadIdx.x & 31) == 0) warp_ss[threadIdx.x / 32] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        for (int w = 0; w < kEmbedThreads / 32; ++w) tot = __fadd_rn(tot, warp_ss[w]);
        s_rstd = rsqrtf(__fadd_rn(__fdiv_rn(tot, (float)n_embd), eps));
    }
    __syncthreads();
    const float rstd = s_rstd;
    for (uint32_t i = threadIdx.x; i < chunks; i += kEmbedThreads) {
        const uint4 v = src[i], w = norm_w[i];
        const uint32_t u[4] = {v.x, v.y, v.z, v.w}, wu[4] = {w.x, w.y, w.z, w.w};
        uint32_t o[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float lo = __fmul_rn(__fmul_rn(bf2f((uint16_t)(u[k] & 0xffffu)), rstd), bf2f((uint16_t)(wu[k] & 0xffffu)));
            const float hi = __fmul_rn(__fmul_rn(bf2f((uint16_t)(u[k] >> 16)), rstd), bf2f((uint16_t)(wu[k] >> 16)));
            o[k] = (uint32_t)f2bf(lo) | ((uint32_t)f2bf(hi) << 16);
        }
        dst[i] = make_uint4(o[0], o[1], o[2], o[3]);
    }
}

/** Logits finish in place: columns < n_real are multiplied by `scale`, padding columns [n_real, n_logits) are set
 *  to -inf.  Grid (ceil(n_logits / (kFinishThreads * 8)), n rows); block kFinishThreads; grid-stride over columns.
 *  @param logits  device f32 rows, row r at logits + r * ld
 *  @param ld      row pitch in elements
 *  @param n_logits columns to touch (the padded vocab)
 *  @param n_real  real-id columns (ids >= n_real are padding)
 *  @param scale   the family's logit scale */
__global__ void __launch_bounds__(kFinishThreads)
head_finish_kernel(float *__restrict__ logits, uint64_t ld, uint32_t n_logits, uint32_t n_real, float scale) {
    float *row = logits + (uint64_t)blockIdx.y * ld;
    for (uint32_t c = blockIdx.x * kFinishThreads + threadIdx.x; c < n_logits; c += gridDim.x * kFinishThreads)
        row[c] = c < n_real ? __fmul_rn(row[c], scale) : -INFINITY;
}

/** Validates a head declaration (width a positive multiple of 8, 1 <= n_real <= n_logits, finite nonzero scale,
 *  16-byte aligned norm weight with eps >= 0); names the refusal on stderr.
 *  @param h   the declaration (may be NULL: refused)
 *  @param who the caller's name for the message
 *  @return true when the declaration is usable */
bool decl_ok(const pulsar_head_decl *h, const char *who) {
    if (!h || h->n_embd == 0 || h->n_embd % 8 || h->n_logits == 0 || h->n_real == 0 || h->n_real > h->n_logits ||
        !isfinite(h->logit_scale) || h->logit_scale == 0.0f || (h->embed_norm_w && !(h->embed_norm_eps >= 0.0f)) ||
        ((uintptr_t)h->embed_norm_w & 15u)) {
        fprintf(stderr, "pulsar: head %s: a declaration with n_embd %u (a positive multiple of 8), n_real %u of %u logits, "
                        "scale %g (finite, nonzero) or a misaligned / negative-eps norm -- refusing\n",
                who, h ? h->n_embd : 0, h ? h->n_real : 0, h ? h->n_logits : 0, h ? (double)h->logit_scale : 0.0);
        return false;
    }
    return true;
}

/** Checks the last launch.
 *  @param who the launch's name for the message
 *  @return 0, or -3 after naming the CUDA error on stderr */
int launched(const char *who) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "pulsar: head %s: launch failed: %s\n", who, cudaGetErrorString(e));
        return -3;
    }
    return 0;
}

} // namespace

extern "C" int pulsar_head_embed_launch(const pulsar_head_decl *h, const uint16_t *table, uint32_t n_vocab, const int32_t *ids,
                                        uint32_t n, uint16_t *out, uint64_t ld_out, cudaStream_t stream) {
    if (!decl_ok(h, "embed")) return -1;
    if (!table || !ids || !out || n == 0 || n_vocab == 0 || ld_out < h->n_embd || ld_out % 8 ||
        (((uintptr_t)table | (uintptr_t)out) & 15u)) {
        fprintf(stderr, "pulsar: head embed: no table / ids / output, no rows, or a misaligned table / output / ld_out "
                        "-- refusing\n");
        return -1;
    }
    head_embed_kernel<<<n, kEmbedThreads, 0, stream>>>((const uint4 *)table, n_vocab, ids, h->n_embd,
                                                       (const uint4 *)h->embed_norm_w, h->embed_norm_eps, out, ld_out);
    return launched("embed");
}

extern "C" int pulsar_head_logits_finish_launch(const pulsar_head_decl *h, float *logits, uint32_t n, uint64_t ld,
                                                cudaStream_t stream) {
    if (!decl_ok(h, "logits")) return -1;
    if (!logits || n == 0 || n > 65535u || ld < h->n_logits) {
        fprintf(stderr, "pulsar: head logits: no logits, %u rows (1 .. 65535), or ld %llu below %u logits -- refusing\n", n,
                (unsigned long long)ld, h->n_logits);
        return -1;
    }
    if (h->logit_scale == 1.0f && h->n_real == h->n_logits) return 0;   /* nothing declared */
    const uint32_t blocks = (h->n_logits + kFinishThreads * 8 - 1) / (kFinishThreads * 8);
    head_finish_kernel<<<dim3(blocks, n), kFinishThreads, 0, stream>>>(logits, ld, h->n_logits, h->n_real, h->logit_scale);
    return launched("logits");
}

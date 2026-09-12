/* L216 -- DeepSeek-V4-Flash-Vision-Exp ViT + aligner forward.
 *
 * A CORRECTNESS-FIRST implementation: every kernel here is the obvious one, not
 * the fast one.  The tower runs once per image (not per token), and the first
 * job is to agree with the reference -- tests/vision_tower_gate.cpp grades the
 * aligner output against stage dumps from the checkpoint's own
 * inference/vision.py.  Once it agrees, the GEMMs are the obvious thing to
 * replace with the engine's bf16 matmul.
 *
 * Fidelity notes -- each is what the reference does, and each is a place a
 * plausible implementation goes wrong:
 *   - RMSNorm: eps 1e-6, mean of squares, weight applied in f32 then cast to
 *     bf16 (`(self.weight * x).to(dtype)` with x already normalized f32).
 *   - Every Linear accumulates in f32 and rounds once to bf16; bias is f32.
 *   - 2D RoPE: rope_dim = head_dim/2 = 32.  inv_freq has 16 entries over that
 *     32; the flattened frequency vector is [row * inv_freq | col * inv_freq],
 *     so pair i uses inv_freq[i % 16], with row for i < 16 and col for i >= 16.
 *     apply_rotary chunks the 64-wide head into two 32-wide halves:
 *       out[i]      = x1[i]*cos[i] - x2[i]*sin[i]
 *       out[32 + i] = x2[i]*cos[i] + x1[i]*sin[i]
 *   - Attention is BIDIRECTIONAL (no mask), scale 1/sqrt(head_dim), f32 softmax.
 *   - The MLP is bias-free SwiGLU: gate = w1(x)[:, :inter], up = w1(x)[:, inter:].
 *   - The aligner zero-pads the (n_h, n_w) grid up to a multiple of the
 *     downsample ratio and unfolds r x r blocks in row-major order with the
 *     CHANNEL outermost (F.unfold's layout), then w1+bias -> GELU (exact/erf)
 *     -> w2+bias.
 *
 * Weights are the artifact's bf16 `vision.*`/`aligner.*` tensors, reached as
 * file offsets into the mapped model -- the engine builds pulsar_vision_offsets
 * because this TU cannot see engine types (see pulsar_gpu.h). */
#include "pulsar_cuda_internal.h"

#include <cuda_bf16.h>
#include <math.h>

typedef __nv_bfloat16 bf16;

static __device__ __forceinline__ float b2f(bf16 v) { return __bfloat162float(v); }
static __device__ __forceinline__ bf16 f2b(float v) { return __float2bfloat16_rn(v); }

/* rows x dim, one block per row, block-wide reduction of the sum of squares. */
__global__ static void vk_rmsnorm(const bf16 *__restrict__ x, const bf16 *__restrict__ w,
                                  bf16 *__restrict__ y, int dim, float eps) {
    extern __shared__ float ssum[];
    const int row = blockIdx.x;
    const bf16 *xr = x + (size_t)row * dim;
    bf16 *yr = y + (size_t)row * dim;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = b2f(xr[i]);
        acc += v * v;
    }
    ssum[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) ssum[threadIdx.x] += ssum[threadIdx.x + s];
        __syncthreads();
    }
    const float rstd = rsqrtf(ssum[0] / (float)dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        yr[i] = f2b(b2f(w[i]) * (b2f(xr[i]) * rstd));
}

/* y[m,n] = sum_k x[m,k] * W[n,k] (+ bias[n]), W is [N,K] row-major -- the
 * GGUF's [K,N] with K contiguous is the same bytes. */
__global__ static void vk_linear(const bf16 *__restrict__ x, const bf16 *__restrict__ W,
                                 const bf16 *__restrict__ bias, bf16 *__restrict__ y,
                                 int K, int N) {
    const int m = blockIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const bf16 *xr = x + (size_t)m * K;
    const bf16 *wr = W + (size_t)n * K;
    float acc = bias ? b2f(bias[n]) : 0.0f;
    for (int k = 0; k < K; k++) acc += b2f(xr[k]) * b2f(wr[k]);
    y[(size_t)m * N + n] = f2b(acc);
}

/* SwiGLU over a fused [gate | up] row of width 2*inter. */
__global__ static void vk_silu_mul(const bf16 *__restrict__ gu, bf16 *__restrict__ y,
                                   int inter) {
    const int m = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= inter) return;
    const float g = b2f(gu[(size_t)m * 2 * inter + i]);
    const float u = b2f(gu[(size_t)m * 2 * inter + inter + i]);
    y[(size_t)m * inter + i] = f2b((g / (1.0f + expf(-g))) * u);
}

/* a += b, in f32 then rounded (torch's bf16 add). */
__global__ static void vk_add(bf16 *__restrict__ a, const bf16 *__restrict__ b, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = f2b(b2f(a[i]) + b2f(b[i]));
}

/* 2D RoPE over q and k in place: (n_tok, n_heads, head_dim).  One block per
 * (token, head), one thread per frequency pair. */
__global__ static void vk_rope2d(bf16 *__restrict__ q, bf16 *__restrict__ k,
                                 int n_heads, int head_dim, int n_w,
                                 int rope_dim, float theta) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int i = threadIdx.x;                     /* pair index in [0, rope_dim) */
    const int half = rope_dim;                     /* each head half is this wide */
    const int nfreq = rope_dim / 2;                /* 16 frequencies per axis */
    if (i >= half) return;
    const int j = i % nfreq;
    const float inv = 1.0f / powf(theta, (float)(2 * j) / (float)rope_dim);
    const float pos = (i < nfreq) ? (float)(t / n_w) : (float)(t % n_w);
    const float c = cosf(pos * inv), s = sinf(pos * inv);
    const size_t base = ((size_t)t * n_heads + h) * head_dim;
    for (int which = 0; which < 2; which++) {
        bf16 *p = which ? k : q;
        const float x1 = b2f(p[base + i]);
        const float x2 = b2f(p[base + half + i]);
        p[base + i] = f2b(x1 * c - x2 * s);
        p[base + half + i] = f2b(x2 * c + x1 * s);
    }
}

/* Bidirectional attention, one block per (head, query token), one thread per
 * head dimension, running-max softmax in f32. */
__global__ static void vk_attention(const bf16 *__restrict__ q, const bf16 *__restrict__ k,
                                    const bf16 *__restrict__ v, bf16 *__restrict__ o,
                                    int n_tok, int n_heads, int head_dim) {
    extern __shared__ float red[];
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int d = threadIdx.x;
    const float scale = rsqrtf((float)head_dim);
    const size_t qbase = ((size_t)t * n_heads + h) * head_dim;
    const float qv = b2f(q[qbase + d]);
    float m = -INFINITY, l = 0.0f, acc = 0.0f;
    for (int j = 0; j < n_tok; j++) {
        const size_t kbase = ((size_t)j * n_heads + h) * head_dim;
        red[d] = qv * b2f(k[kbase + d]);
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if ((int)threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
            __syncthreads();
        }
        const float sc = red[0] * scale;
        __syncthreads();
        const float vv = b2f(v[kbase + d]);
        if (sc > m) {
            const float alpha = expf(m - sc);
            l = l * alpha + 1.0f;
            acc = acc * alpha + vv;
            m = sc;
        } else {
            const float p = expf(sc - m);
            l += p;
            acc += p * vv;
        }
        __syncthreads();
    }
    o[qbase + d] = f2b(acc / l);
}

/* Aligner gather: zero-pad the (n_h, n_w) grid up to a multiple of r and unfold
 * r x r blocks with the channel outermost -- F.unfold's layout.  One thread per
 * element of (n_llm_rows, dim*r*r). */
__global__ static void vk_aligner_gather(const bf16 *__restrict__ x, bf16 *__restrict__ out,
                                         int n_h, int n_w, int dim, int r) {
    const int blocks_w = (n_w + r - 1) / r;
    const int blocks_h = (n_h + r - 1) / r;
    const size_t total = (size_t)blocks_h * blocks_w * dim * r * r;
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int kk = (int)(idx % (size_t)(r * r));
    size_t rest = idx / (size_t)(r * r);
    const int c = (int)(rest % (size_t)dim);
    rest /= (size_t)dim;
    const int bw = (int)(rest % (size_t)blocks_w);
    const int bh = (int)(rest / (size_t)blocks_w);
    const int hh = bh * r + kk / r, ww = bw * r + kk % r;
    out[idx] = (hh < n_h && ww < n_w) ? x[((size_t)hh * n_w + ww) * dim + c] : f2b(0.0f);
}

/* GELU, exact (erf) form -- F.gelu's default -- in f32 then rounded. */
__global__ static void vk_gelu(bf16 *__restrict__ y, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = b2f(y[i]);
        y[i] = f2b(0.5f * v * (1.0f + erff(v * 0.70710678118654752440f)));
    }
}

#define VK_THREADS 256

int pulsar_cuda_vision_forward(const pulsar_vision_offsets *o,
                               const void *map, uint64_t map_size,
                               const uint16_t *patches_host, int n_h, int n_w,
                               uint16_t *out_host, int out_cap, int *out_rows) {
    (void)map_size;
    if (!o || !map || o->n_layers != PULSAR_VISION_LAYERS) return 0;
    const int P = (int)PULSAR_VISION_PATCH;
    const int D = (int)PULSAR_VISION_DIM;
    const int H = (int)PULSAR_VISION_HEADS;
    const int I = (int)PULSAR_VISION_INTER;
    const int R = (int)PULSAR_VISION_DOWNSAMPLE;
    const int T = (int)o->text_dim;
    const int head_dim = D / H;
    const int rope_dim = head_dim / 2;
    const int n_tok = n_h * n_w;
    const int n_llm = ((n_h + R - 1) / R) * ((n_w + R - 1) / R);
    const size_t x_elems = (size_t)n_tok * D;
    const size_t alg_elems = (size_t)n_llm * D * R * R;
    const size_t out_elems = (size_t)n_llm * T;
    const size_t smem_norm = (size_t)((D + VK_THREADS - 1) / VK_THREADS * VK_THREADS) * sizeof(float);
    const float eps = 1e-6f;

    void *d_patches = NULL, *d_x = NULL, *d_tmp = NULL, *d_qkv = NULL, *d_attn = NULL,
         *d_mlp = NULL, *d_normed = NULL, *d_alg = NULL, *d_alg2 = NULL, *d_out = NULL;
    int ok = 0;

#define WT(off) ((const bf16 *)(const void *)((const char *)map + (off)))
#define CUDA_ALLOC(p, bytes) do { if (cudaMalloc(&(p), (bytes)) != cudaSuccess) goto done; } while (0)
#define CUDA_LAUNCH(what) do { if (!cuda_ok(cudaGetLastError(), what)) goto done; } while (0)

    CUDA_ALLOC(d_patches, (size_t)n_tok * 3 * P * P * sizeof(bf16));
    CUDA_ALLOC(d_x, x_elems * sizeof(bf16));
    CUDA_ALLOC(d_tmp, x_elems * sizeof(bf16));
    CUDA_ALLOC(d_attn, x_elems * sizeof(bf16));
    CUDA_ALLOC(d_normed, x_elems * sizeof(bf16));
    CUDA_ALLOC(d_qkv, (size_t)n_tok * 3 * D * sizeof(bf16));
    CUDA_ALLOC(d_mlp, (size_t)n_tok * 2 * I * sizeof(bf16));
    CUDA_ALLOC(d_alg, alg_elems * sizeof(bf16));
    CUDA_ALLOC(d_alg2, out_elems * sizeof(bf16));
    CUDA_ALLOC(d_out, out_elems * sizeof(bf16));
    if (cudaMemcpy(d_patches, patches_host, (size_t)n_tok * 3 * P * P * sizeof(bf16),
                   cudaMemcpyHostToDevice) != cudaSuccess) goto done;

    vk_linear<<<dim3((unsigned)((D + VK_THREADS - 1) / VK_THREADS), (unsigned)n_tok), VK_THREADS>>>(
        (const bf16 *)d_patches, WT(o->patch_proj), WT(o->patch_bias), (bf16 *)d_x, 3 * P * P, D);
    CUDA_LAUNCH("vision patch_embed");

    for (uint32_t li = 0; li < o->n_layers; li++) {
        vk_rmsnorm<<<(unsigned)n_tok, VK_THREADS, smem_norm>>>(
            (const bf16 *)d_x, WT(o->block[li].norm1), (bf16 *)d_tmp, D, eps);
        CUDA_LAUNCH("vision norm1");

        vk_linear<<<dim3((unsigned)((3 * D + VK_THREADS - 1) / VK_THREADS), (unsigned)n_tok), VK_THREADS>>>(
            (const bf16 *)d_tmp, WT(o->block[li].wqkv), WT(o->block[li].wqkv_bias),
            (bf16 *)d_qkv, D, 3 * D);
        CUDA_LAUNCH("vision wqkv");

        {
            bf16 *q = (bf16 *)d_qkv;
            bf16 *kk = q + (size_t)n_tok * D;
            bf16 *vv = kk + (size_t)n_tok * D;
            vk_rope2d<<<dim3((unsigned)n_tok, H), (unsigned)rope_dim>>>(
                q, kk, H, head_dim, n_w, rope_dim, (float)PULSAR_VISION_ROPE_THETA);
            CUDA_LAUNCH("vision rope2d");
            vk_attention<<<dim3(H, (unsigned)n_tok), (unsigned)head_dim, (size_t)head_dim * sizeof(float)>>>(
                q, kk, vv, (bf16 *)d_attn, n_tok, H, head_dim);
            CUDA_LAUNCH("vision attention");
        }

        vk_linear<<<dim3((unsigned)((D + VK_THREADS - 1) / VK_THREADS), (unsigned)n_tok), VK_THREADS>>>(
            (const bf16 *)d_attn, WT(o->block[li].wo), WT(o->block[li].wo_bias),
            (bf16 *)d_tmp, D, D);
        CUDA_LAUNCH("vision wo");
        vk_add<<<(unsigned)((x_elems + VK_THREADS - 1) / VK_THREADS), VK_THREADS>>>(
            (bf16 *)d_x, (const bf16 *)d_tmp, x_elems);
        CUDA_LAUNCH("vision attn residual");

        vk_rmsnorm<<<(unsigned)n_tok, VK_THREADS, smem_norm>>>(
            (const bf16 *)d_x, WT(o->block[li].norm2), (bf16 *)d_tmp, D, eps);
        CUDA_LAUNCH("vision norm2");

        vk_linear<<<dim3((unsigned)((2 * I + VK_THREADS - 1) / VK_THREADS), (unsigned)n_tok), VK_THREADS>>>(
            (const bf16 *)d_tmp, WT(o->block[li].w1), NULL, (bf16 *)d_mlp, D, 2 * I);
        CUDA_LAUNCH("vision mlp w1");

        vk_silu_mul<<<dim3((unsigned)((I + VK_THREADS - 1) / VK_THREADS), (unsigned)n_tok), VK_THREADS>>>(
            (const bf16 *)d_mlp, (bf16 *)d_tmp, I);
        CUDA_LAUNCH("vision silu_mul");

        vk_linear<<<dim3((unsigned)((D + VK_THREADS - 1) / VK_THREADS), (unsigned)n_tok), VK_THREADS>>>(
            (const bf16 *)d_tmp, WT(o->block[li].w2), NULL, (bf16 *)d_attn, I, D);
        CUDA_LAUNCH("vision mlp w2");
        vk_add<<<(unsigned)((x_elems + VK_THREADS - 1) / VK_THREADS), VK_THREADS>>>(
            (bf16 *)d_x, (const bf16 *)d_attn, x_elems);
        CUDA_LAUNCH("vision mlp residual");
    }

    vk_rmsnorm<<<(unsigned)n_tok, VK_THREADS, smem_norm>>>(
        (const bf16 *)d_x, WT(o->norm), (bf16 *)d_normed, D, eps);
    CUDA_LAUNCH("vision final norm");

    vk_aligner_gather<<<(unsigned)((alg_elems + VK_THREADS - 1) / VK_THREADS), VK_THREADS>>>(
        (const bf16 *)d_normed, (bf16 *)d_alg, n_h, n_w, D, R);
    CUDA_LAUNCH("vision aligner gather");

    vk_linear<<<dim3((unsigned)((T + VK_THREADS - 1) / VK_THREADS), (unsigned)n_llm), VK_THREADS>>>(
        (const bf16 *)d_alg, WT(o->aligner_w1), WT(o->aligner_b1), (bf16 *)d_alg2, D * R * R, T);
    CUDA_LAUNCH("vision aligner w1");

    vk_gelu<<<(unsigned)((out_elems + VK_THREADS - 1) / VK_THREADS), VK_THREADS>>>(
        (bf16 *)d_alg2, out_elems);
    CUDA_LAUNCH("vision aligner gelu");

    vk_linear<<<dim3((unsigned)((T + VK_THREADS - 1) / VK_THREADS), (unsigned)n_llm), VK_THREADS>>>(
        (const bf16 *)d_alg2, WT(o->aligner_w2), WT(o->aligner_b2), (bf16 *)d_out, T, T);
    CUDA_LAUNCH("vision aligner w2");

    if (out_elems > (size_t)out_cap) goto done;
    if (cudaMemcpy(out_host, d_out, out_elems * sizeof(bf16), cudaMemcpyDeviceToHost) != cudaSuccess)
        goto done;
    *out_rows = n_llm;
    ok = 1;

done:
    cudaFree(d_patches); cudaFree(d_x); cudaFree(d_tmp); cudaFree(d_attn); cudaFree(d_normed);
    cudaFree(d_qkv); cudaFree(d_mlp); cudaFree(d_alg); cudaFree(d_alg2); cudaFree(d_out);
    return ok;
#undef CUDA_LAUNCH
#undef CUDA_ALLOC
#undef WT
}

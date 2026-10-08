/* L216 -- DeepSeek-V4-Flash-Vision-Exp ViT + aligner forward; L268 -- Qwen3.8-Flash-Next's tower + merger over
 * the same kernels (pulsar_cuda_qwen_vision_forward, below).  The two towers share every primitive their
 * architectures share -- linear + bias, residual add, the q|k|v split, the NeoX-halves 2D RoPE (per-token positions:
 * DeepSeek's raster, Qwen's merge-block order), bidirectional attention, erf-GELU -- and add only their own:
 * DeepSeek's RMSNorm / SwiGLU / aligner unfold, Qwen's LayerNorm / tanh-GELU / interpolated learned positions /
 * 2x2 merger.
 *
 * A CORRECTNESS-FIRST implementation: every elementwise kernel here is the obvious one.  L268: the linears and
 * the attention run on cuBLASLt (pulsar_cuda_vision_gemm: f32 accumulate, one rounding), for both towers.  The tower runs once per image (not per token), and the first
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

/* wqkv's output is (n_tok, 3D) with q|k|v ADJACENT per token -- the reference
 * does self.wqkv(x).chunk(3, dim=-1), so token t's keys are at t*3D + D and its
 * values at t*3D + 2D.  Rope and attention want three planes, and reading the
 * buffer as planes is what made block0 grossly wrong (patch_embed was exact, so
 * the weights were fine; the very first thing after them was not). */
__global__ static void vk_split_qkv(const bf16 *__restrict__ qkv,
                                    bf16 *__restrict__ q, bf16 *__restrict__ k,
                                    bf16 *__restrict__ v, int D, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const size_t t = i / (size_t)D, d = i % (size_t)D;
    q[i] = qkv[t * 3 * (size_t)D + d];
    k[i] = qkv[t * 3 * (size_t)D + D + d];
    v[i] = qkv[t * 3 * (size_t)D + 2 * (size_t)D + d];
}

/* 2D RoPE over q and k in place: (n_tok, n_heads, head_dim).  One block per
 * (token, head), one thread per frequency pair.  The head is two halves of
 * `rope_dim` (= head_dim / 2) rotated against each other; the first rope_dim / 2
 * pairs take the token's ROW, the rest its COLUMN, frequency index i mod
 * rope_dim / 2 -- DeepSeek's layout and Qwen's (cat(f_h, f_w) twice) alike.
 * `pos` is (row, col) per token: the caller's patch order decides them. */
__global__ static void vk_rope2d(bf16 *__restrict__ q, bf16 *__restrict__ k,
                                 int n_heads, int head_dim, const int2 *__restrict__ pos2,
                                 int rope_dim, float theta) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int i = threadIdx.x;                     /* pair index in [0, rope_dim) */
    const int half = rope_dim;                     /* each head half is this wide */
    const int nfreq = rope_dim / 2;                /* frequencies per axis */
    if (i >= half) return;
    const int j = i % nfreq;
    const float inv = 1.0f / powf(theta, (float)(2 * j) / (float)rope_dim);
    const float pos = (i < nfreq) ? (float)pos2[t].x : (float)pos2[t].y;
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


/* LayerNorm with bias (torch's, eps given): mean and biased variance in f32 over
 * the row, y = (x - mean) * rstd * w + b in f32, rounded once.  One block per
 * row, block-wide reductions (blockDim a power of two). */
__global__ static void vk_layernorm(const bf16 *__restrict__ x, const bf16 *__restrict__ w,
                                    const bf16 *__restrict__ b, bf16 *__restrict__ y, int dim, float eps) {
    extern __shared__ float red[];
    const int row = blockIdx.x;
    const bf16 *xr = x + (size_t)row * dim;
    bf16 *yr = y + (size_t)row * dim;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) acc += b2f(xr[i]);
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float mean = red[0] / (float)dim;
    __syncthreads();
    acc = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float d = b2f(xr[i]) - mean;
        acc += d * d;
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float rstd = rsqrtf(red[0] / (float)dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        yr[i] = f2b((b2f(xr[i]) - mean) * rstd * b2f(w[i]) + b2f(b[i]));
}

/* GELU, tanh approximation (gelu_pytorch_tanh) -- in f32 then rounded. */
__global__ static void vk_gelu_tanh(bf16 *__restrict__ y, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = b2f(y[i]);
        const float inner = 0.79788456080286535588f * (v + 0.044715f * v * v * v);
        y[i] = f2b(0.5f * v * (1.0f + tanhf(inner)));
    }
}

/* Qwen's learned positions, resampled to the patch grid: per patch, the bf16
 * table rows at its 4 bilinear taps times their f32 weights, summed in f32 (tap
 * order), rounded to bf16 and added to the patch embedding (a bf16 add).  The
 * taps and weights are the host's (HF's formula, align_corners). */
__global__ static void vk_pos_embed_add(bf16 *__restrict__ x, const bf16 *__restrict__ table,
                                        const int32_t *__restrict__ idx, const float *__restrict__ wt,
                                        int dim, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const size_t row = i / (size_t)dim, c = i % (size_t)dim;
    float s = 0.0f;
    for (int t = 0; t < 4; t++) s += b2f(table[(size_t)idx[row * 4 + t] * dim + c]) * wt[row * 4 + t];
    x[i] = f2b(b2f(x[i]) + b2f(f2b(s)));
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

/* L268 (both towers): y bf16 [rows x N] = x bf16 [rows x K] . W^T (+ bias) -- the GEMM accumulates in f32
 * (pulsar_cuda_vision_gemm, cuBLASLt), then one kernel adds the bias in f32 and rounds once to bf16: what every
 * reference Linear does.  `f32` is the caller's scratch of at least rows x N floats. */
__global__ static void vk_bias_round(const float *__restrict__ acc, const bf16 *__restrict__ bias,
                                     bf16 *__restrict__ y, int N, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = f2b(acc[i] + (bias ? b2f(bias[i % (size_t)N]) : 0.0f));
}

static bool vlinear(const void *x, const void *W, const void *bias, void *y, int K, int N, int rows, float *f32) {
    if (!pulsar_cuda_vision_gemm(f32, N, (const uint16_t *)x, K, (const uint16_t *)W, K, 1, rows, N, K)) return false;
    const size_t n = (size_t)rows * N;
    vk_bias_round<<<(unsigned)((n + VK_THREADS - 1) / VK_THREADS), VK_THREADS>>>(f32, (const bf16 *)bias, (bf16 *)y,
                                                                                N, n);
    return cuda_ok(cudaGetLastError(), "vision bias_round");
}

/* Softmax over each row of `s` (f32 [rows x n] scores) after scaling, written as bf16 probabilities (the second
 * GEMM's operand, as flash attention rounds them).  One block per row. */
__global__ static void vk_softmax_rows(const float *__restrict__ s, bf16 *__restrict__ p, int n, float scale) {
    extern __shared__ float red[];
    const float *sr = s + (size_t)blockIdx.x * n;
    bf16 *pr = p + (size_t)blockIdx.x * n;
    float m = -INFINITY;
    for (int j = threadIdx.x; j < n; j += blockDim.x) m = fmaxf(m, sr[j] * scale);
    red[threadIdx.x] = m;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if ((int)threadIdx.x < k) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + k]);
        __syncthreads();
    }
    m = red[0];
    __syncthreads();
    float sum = 0.0f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) sum += expf(sr[j] * scale - m);
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if ((int)threadIdx.x < k) red[threadIdx.x] += red[threadIdx.x + k];
        __syncthreads();
    }
    const float inv = 1.0f / red[0];
    for (int j = threadIdx.x; j < n; j += blockDim.x) pr[j] = f2b(expf(sr[j] * scale - m) * inv);
}

/* One head's f32 output rows [rows x hd] into the attention output (bf16 [n_tok][heads][hd]) at row0. */
__global__ static void vk_store_head(const float *__restrict__ o, bf16 *__restrict__ out, int row0, int rows,
                                     int heads, int head, int hd) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * hd) return;
    const size_t r = i / (size_t)hd, d = i % (size_t)hd;
    out[((size_t)(row0 + r) * heads + head) * hd + d] = f2b(o[i]);
}

/* Bidirectional attention over one image (q, k, v, out: bf16 [n_tok][heads][hd]), per head and per query chunk:
 * S = Q K^T in f32 on the GEMM, a scaled row softmax to bf16 P, O = P V in f32, rounded once.  The chunk keeps the
 * score block within `s_bytes` (`s` f32 and `p` bf16 scratch of that many score entries; `o` f32 chunk x hd). */
static bool vattention(const void *q, const void *k, const void *v, void *out, int n_tok, int heads, int hd,
                       float *s, bf16 *p, float *o, size_t s_entries) {
    const int ld = heads * hd;
    const float scale = 1.0f / sqrtf((float)hd);
    int chunk = (int)(s_entries / (size_t)n_tok);
    if (chunk > n_tok) chunk = n_tok;
    if (chunk < 1) return false;
    for (int h = 0; h < heads; h++) {
        const uint16_t *qh = (const uint16_t *)q + (size_t)h * hd, *kh = (const uint16_t *)k + (size_t)h * hd,
                       *vh = (const uint16_t *)v + (size_t)h * hd;
        for (int r0 = 0; r0 < n_tok; r0 += chunk) {
            const int rows = n_tok - r0 < chunk ? n_tok - r0 : chunk;
            if (!pulsar_cuda_vision_gemm(s, n_tok, qh + (size_t)r0 * ld, ld, kh, ld, 1, rows, n_tok, hd)) return false;
            vk_softmax_rows<<<(unsigned)rows, VK_THREADS, VK_THREADS * sizeof(float)>>>(s, p, n_tok, scale);
            if (!cuda_ok(cudaGetLastError(), "vision softmax")) return false;
            if (!pulsar_cuda_vision_gemm(o, hd, (const uint16_t *)p, n_tok, vh, ld, 0, rows, hd, n_tok)) return false;
            vk_store_head<<<(unsigned)(((size_t)rows * hd + VK_THREADS - 1) / VK_THREADS), VK_THREADS>>>(
                o, (bf16 *)out, r0, rows, heads, h, hd);
            if (!cuda_ok(cudaGetLastError(), "vision store_head")) return false;
        }
    }
    return true;
}

/* The attention scratch: score entries per chunk (f32 S + bf16 P), at most 256 Mi entries (1 GiB of f32). */
static size_t vattention_entries(int n_tok) {
    const size_t cap = (size_t)256 << 20, full = (size_t)n_tok * n_tok;
    return full < cap ? full : cap;
}

int pulsar_cuda_vision_forward(const pulsar_vision_offsets *o,
                               const void *map, uint64_t map_size,
                               const uint16_t *patches_host, int n_h, int n_w,
                               uint16_t *out_host, int out_cap, int *out_rows,
                               uint16_t *dbg, uint32_t dbg_blocks) {
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

    void *d_patches = NULL, *d_x = NULL, *d_tmp = NULL, *d_mid = NULL, *d_qkv = NULL,
         *d_q = NULL, *d_k = NULL, *d_v = NULL, *d_attn = NULL, *d_mlp = NULL,
         *d_normed = NULL, *d_alg = NULL, *d_alg2 = NULL, *d_out = NULL, *d_pos = NULL;
    void *d_f32 = NULL, *d_s = NULL, *d_p = NULL, *d_o = NULL;   /* L268: the GEMM scratch, the attention's */
    const size_t s_entries = vattention_entries(n_tok);
    size_t f32_elems = (size_t)n_tok * (size_t)(3 * D > 2 * I ? 3 * D : 2 * I);
    if ((size_t)n_llm * T > f32_elems) f32_elems = (size_t)n_llm * T;
    int ok = 0;

#define WT(off) ((const bf16 *)(const void *)((const char *)map + (off)))
#define CUDA_ALLOC(p, bytes) do { if (cudaMalloc(&(p), (bytes)) != cudaSuccess) goto done; } while (0)
#define CUDA_LAUNCH(what) do { if (!cuda_ok(cudaGetLastError(), what)) goto done; } while (0)
/* The optional stage dump: `row` indexes a (2 + dbg_blocks) x n_tok*D block. */
#define DUMP_STAGE(row, src) do { \
        if (dbg && cudaMemcpy((uint16_t *)dbg + (size_t)(row) * x_elems, (src), \
                              x_elems * sizeof(bf16), cudaMemcpyDeviceToHost) != cudaSuccess) \
            goto done; \
    } while (0)

    CUDA_ALLOC(d_patches, (size_t)n_tok * 3 * P * P * sizeof(bf16));
    CUDA_ALLOC(d_x, x_elems * sizeof(bf16));
    CUDA_ALLOC(d_tmp, x_elems * sizeof(bf16));
    /* the SwiGLU result is inter-wide (2816), NOT dim-wide: writing it into
     * d_tmp overran that buffer by 2.75x and corrupted the residual stream. */
    CUDA_ALLOC(d_mid, (size_t)n_tok * I * sizeof(bf16));
    CUDA_ALLOC(d_attn, x_elems * sizeof(bf16));
    CUDA_ALLOC(d_normed, x_elems * sizeof(bf16));
    CUDA_ALLOC(d_qkv, (size_t)n_tok * 3 * D * sizeof(bf16));
    CUDA_ALLOC(d_q, x_elems * sizeof(bf16));
    CUDA_ALLOC(d_k, x_elems * sizeof(bf16));
    CUDA_ALLOC(d_v, x_elems * sizeof(bf16));
    CUDA_ALLOC(d_mlp, (size_t)n_tok * 2 * I * sizeof(bf16));
    CUDA_ALLOC(d_alg, alg_elems * sizeof(bf16));
    CUDA_ALLOC(d_alg2, out_elems * sizeof(bf16));
    CUDA_ALLOC(d_out, out_elems * sizeof(bf16));
    CUDA_ALLOC(d_f32, f32_elems * sizeof(float));
    CUDA_ALLOC(d_s, s_entries * sizeof(float));
    CUDA_ALLOC(d_p, s_entries * sizeof(bf16));
    CUDA_ALLOC(d_o, (s_entries / (size_t)n_tok + 1) * (size_t)head_dim * sizeof(float));
    if (cudaMemcpy(d_patches, patches_host, (size_t)n_tok * 3 * P * P * sizeof(bf16),
                   cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    {
        /* the rope's (row, col) per token: DeepSeek's patches are raster order */
        int2 *pos = (int2 *)malloc((size_t)n_tok * sizeof(int2));
        if (!pos) goto done;
        for (int t = 0; t < n_tok; t++) pos[t] = make_int2(t / n_w, t % n_w);
        const bool up = cudaMalloc(&d_pos, (size_t)n_tok * sizeof(int2)) == cudaSuccess &&
                        cudaMemcpy(d_pos, pos, (size_t)n_tok * sizeof(int2), cudaMemcpyHostToDevice) == cudaSuccess;
        free(pos);
        if (!up) goto done;
    }

    if (!vlinear(d_patches, WT(o->patch_proj), WT(o->patch_bias), d_x, 3 * P * P, D, n_tok, (float *)d_f32)) goto done;
    DUMP_STAGE(0, d_x);

    for (uint32_t li = 0; li < o->n_layers; li++) {
        vk_rmsnorm<<<(unsigned)n_tok, VK_THREADS, smem_norm>>>(
            (const bf16 *)d_x, WT(o->block[li].norm1), (bf16 *)d_tmp, D, eps);
        CUDA_LAUNCH("vision norm1");

        if (!vlinear(d_tmp, WT(o->block[li].wqkv), WT(o->block[li].wqkv_bias), d_qkv, D, 3 * D, n_tok,
                     (float *)d_f32)) goto done;

        vk_split_qkv<<<(unsigned)((x_elems + VK_THREADS - 1) / VK_THREADS), VK_THREADS>>>(
            (const bf16 *)d_qkv, (bf16 *)d_q, (bf16 *)d_k, (bf16 *)d_v, D, x_elems);
        CUDA_LAUNCH("vision split_qkv");
        {
            vk_rope2d<<<dim3((unsigned)n_tok, H), (unsigned)rope_dim>>>(
                (bf16 *)d_q, (bf16 *)d_k, H, head_dim, (const int2 *)d_pos, rope_dim,
                (float)PULSAR_VISION_ROPE_THETA);
            CUDA_LAUNCH("vision rope2d");
            if (!vattention(d_q, d_k, d_v, d_attn, n_tok, H, head_dim, (float *)d_s, (bf16 *)d_p, (float *)d_o,
                            s_entries)) goto done;
        }

        if (!vlinear(d_attn, WT(o->block[li].wo), WT(o->block[li].wo_bias), d_tmp, D, D, n_tok, (float *)d_f32))
            goto done;
        vk_add<<<(unsigned)((x_elems + VK_THREADS - 1) / VK_THREADS), VK_THREADS>>>(
            (bf16 *)d_x, (const bf16 *)d_tmp, x_elems);
        CUDA_LAUNCH("vision attn residual");

        vk_rmsnorm<<<(unsigned)n_tok, VK_THREADS, smem_norm>>>(
            (const bf16 *)d_x, WT(o->block[li].norm2), (bf16 *)d_tmp, D, eps);
        CUDA_LAUNCH("vision norm2");

        if (!vlinear(d_tmp, WT(o->block[li].w1), NULL, d_mlp, D, 2 * I, n_tok, (float *)d_f32)) goto done;

        vk_silu_mul<<<dim3((unsigned)((I + VK_THREADS - 1) / VK_THREADS), (unsigned)n_tok), VK_THREADS>>>(
            (const bf16 *)d_mlp, (bf16 *)d_mid, I);
        CUDA_LAUNCH("vision silu_mul");

        if (!vlinear(d_mid, WT(o->block[li].w2), NULL, d_attn, I, D, n_tok, (float *)d_f32)) goto done;
        vk_add<<<(unsigned)((x_elems + VK_THREADS - 1) / VK_THREADS), VK_THREADS>>>(
            (bf16 *)d_x, (const bf16 *)d_attn, x_elems);
        CUDA_LAUNCH("vision mlp residual");
        if (li < dbg_blocks) DUMP_STAGE(1 + li, d_x);
    }

    vk_rmsnorm<<<(unsigned)n_tok, VK_THREADS, smem_norm>>>(
        (const bf16 *)d_x, WT(o->norm), (bf16 *)d_normed, D, eps);
    CUDA_LAUNCH("vision final norm");
    DUMP_STAGE(1 + (int)dbg_blocks, d_normed);

    vk_aligner_gather<<<(unsigned)((alg_elems + VK_THREADS - 1) / VK_THREADS), VK_THREADS>>>(
        (const bf16 *)d_normed, (bf16 *)d_alg, n_h, n_w, D, R);
    CUDA_LAUNCH("vision aligner gather");

    if (!vlinear(d_alg, WT(o->aligner_w1), WT(o->aligner_b1), d_alg2, D * R * R, T, n_llm, (float *)d_f32)) goto done;

    vk_gelu<<<(unsigned)((out_elems + VK_THREADS - 1) / VK_THREADS), VK_THREADS>>>(
        (bf16 *)d_alg2, out_elems);
    CUDA_LAUNCH("vision aligner gelu");

    if (!vlinear(d_alg2, WT(o->aligner_w2), WT(o->aligner_b2), d_out, T, T, n_llm, (float *)d_f32)) goto done;

    if (out_elems > (size_t)out_cap) goto done;
    if (cudaMemcpy(out_host, d_out, out_elems * sizeof(bf16), cudaMemcpyDeviceToHost) != cudaSuccess)
        goto done;
    *out_rows = n_llm;
    ok = 1;

done:
    cudaFree(d_patches); cudaFree(d_x); cudaFree(d_tmp); cudaFree(d_mid); cudaFree(d_attn);
    cudaFree(d_normed);
    cudaFree(d_qkv); cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_mlp);
    cudaFree(d_alg); cudaFree(d_alg2); cudaFree(d_out); cudaFree(d_pos);
    cudaFree(d_f32); cudaFree(d_s); cudaFree(d_p); cudaFree(d_o);
    return ok;
#undef DUMP_STAGE
#undef CUDA_LAUNCH
#undef CUDA_ALLOC
#undef WT
}

/* L268: Qwen3.8-Flash-Next's tower + merger over ONE image (HF Qwen4ExpVisionModel, the Qwen3-VL tower without
 * deepstack): patch embed (the Conv3d as a linear over the 1536 patch vector) -> + interpolated learned positions
 * -> 27 x [LayerNorm -> qkv -> 2D RoPE (merge-block positions) -> bidirectional attention -> proj -> residual ->
 * LayerNorm -> fc1 -> tanh-GELU -> fc2 -> residual] -> merger [LayerNorm per patch -> 4 patches (one 2x2 block)
 * concatenated -> fc1 -> erf-GELU -> fc2].  Same correctness-first kernels as DeepSeek's tower, graded the same way
 * (tests/vision_qwen_tower_gate.cpp, the reference's own bf16-vs-fp32 floor). */
int pulsar_cuda_qwen_vision_forward(const pulsar_qwen_vision_weights_dev *w, const uint16_t *patches_host,
                                    const int32_t *pos_host, const int32_t *interp_idx_host,
                                    const float *interp_w_host, int n_tok, uint16_t *out_host, int out_cap,
                                    uint16_t *dbg) {
    if (!w || n_tok <= 0 || n_tok % 4) return 0;
    const int D = (int)PULSAR_QWEN_VISION_DIM, H = (int)PULSAR_QWEN_VISION_HEADS,
              I = (int)PULSAR_QWEN_VISION_INTER, K = (int)PULSAR_QWEN_VISION_PATCH_IN,
              O = (int)PULSAR_QWEN_VISION_OUT, M = 4 * D;
    const int head_dim = D / H, rope_dim = head_dim / 2;
    const int n_out = n_tok / 4;
    const size_t x_elems = (size_t)n_tok * D;
    const unsigned ln_threads = 256;
    const size_t ln_smem = ln_threads * sizeof(float);
    const float eps = 1e-6f;
    void *d_patches = NULL, *d_x = NULL, *d_tmp = NULL, *d_qkv = NULL, *d_q = NULL, *d_k = NULL, *d_v = NULL,
         *d_attn = NULL, *d_mlp = NULL, *d_pos = NULL, *d_idx = NULL, *d_wt = NULL, *d_mid = NULL, *d_out = NULL;
    void *d_f32 = NULL, *d_s = NULL, *d_p = NULL, *d_o = NULL;   /* the GEMM scratch, the attention's */
    const size_t s_entries = vattention_entries(n_tok);
    size_t f32_elems = (size_t)n_tok * (size_t)(3 * D > I ? 3 * D : I);
    if ((size_t)n_out * M > f32_elems) f32_elems = (size_t)n_out * M;
    int ok = 0;
#define WP(p) ((const bf16 *)(p))
#define CUDA_ALLOC(p, bytes) do { if (cudaMalloc(&(p), (bytes)) != cudaSuccess) goto done; } while (0)
#define CUDA_UP(dst, src, bytes) do { if (cudaMemcpy((dst), (src), (bytes), cudaMemcpyHostToDevice) != cudaSuccess) goto done; } while (0)
#define CUDA_LAUNCH(what) do { if (!cuda_ok(cudaGetLastError(), what)) goto done; } while (0)
#define LINEAR(x, W, B, y, k, n, rows) do { \
        if (!vlinear((x), (W), (B), (y), (k), (n), (rows), (float *)d_f32)) goto done; \
    } while (0)
#define DUMP(slot, src) do { \
        if (dbg && cudaMemcpy(dbg + (size_t)(slot) * x_elems, (src), x_elems * sizeof(bf16), \
                              cudaMemcpyDeviceToHost) != cudaSuccess) goto done; \
    } while (0)

    CUDA_ALLOC(d_patches, (size_t)n_tok * K * sizeof(bf16));
    CUDA_ALLOC(d_x, x_elems * sizeof(bf16));
    CUDA_ALLOC(d_tmp, x_elems * sizeof(bf16));
    CUDA_ALLOC(d_attn, x_elems * sizeof(bf16));
    CUDA_ALLOC(d_qkv, x_elems * 3 * sizeof(bf16));
    CUDA_ALLOC(d_q, x_elems * sizeof(bf16));
    CUDA_ALLOC(d_k, x_elems * sizeof(bf16));
    CUDA_ALLOC(d_v, x_elems * sizeof(bf16));
    CUDA_ALLOC(d_mlp, (size_t)n_tok * I * sizeof(bf16));
    CUDA_ALLOC(d_pos, (size_t)n_tok * sizeof(int2));
    CUDA_ALLOC(d_idx, (size_t)n_tok * 4 * sizeof(int32_t));
    CUDA_ALLOC(d_wt, (size_t)n_tok * 4 * sizeof(float));
    CUDA_ALLOC(d_mid, (size_t)n_out * M * sizeof(bf16));
    CUDA_ALLOC(d_out, (size_t)n_out * O * sizeof(bf16));
    CUDA_ALLOC(d_f32, f32_elems * sizeof(float));
    CUDA_ALLOC(d_s, s_entries * sizeof(float));
    CUDA_ALLOC(d_p, s_entries * sizeof(bf16));
    CUDA_ALLOC(d_o, (s_entries / (size_t)n_tok + 1) * (size_t)head_dim * sizeof(float));
    CUDA_UP(d_patches, patches_host, (size_t)n_tok * K * sizeof(bf16));
    CUDA_UP(d_pos, pos_host, (size_t)n_tok * sizeof(int2));
    CUDA_UP(d_idx, interp_idx_host, (size_t)n_tok * 4 * sizeof(int32_t));
    CUDA_UP(d_wt, interp_w_host, (size_t)n_tok * 4 * sizeof(float));

    LINEAR(d_patches, w->patch_w, w->patch_b, d_x, K, D, n_tok);
    vk_pos_embed_add<<<(unsigned)((x_elems + VK_THREADS - 1) / VK_THREADS), VK_THREADS>>>(
        (bf16 *)d_x, WP(w->pos_embed), (const int32_t *)d_idx, (const float *)d_wt, D, x_elems);
    CUDA_LAUNCH("qwen vision pos_embed");
    DUMP(0, d_x);

    for (int li = 0; li < (int)PULSAR_QWEN_VISION_LAYERS; li++) {
        const pulsar_qwen_vision_block_dev *b = &w->block[li];
        vk_layernorm<<<(unsigned)n_tok, ln_threads, ln_smem>>>((const bf16 *)d_x, WP(b->norm1_w), WP(b->norm1_b),
                                                              (bf16 *)d_tmp, D, eps);
        CUDA_LAUNCH("qwen vision norm1");
        LINEAR(d_tmp, b->qkv_w, b->qkv_b, d_qkv, D, 3 * D, n_tok);
        vk_split_qkv<<<(unsigned)((x_elems + VK_THREADS - 1) / VK_THREADS), VK_THREADS>>>(
            (const bf16 *)d_qkv, (bf16 *)d_q, (bf16 *)d_k, (bf16 *)d_v, D, x_elems);
        CUDA_LAUNCH("qwen vision split_qkv");
        vk_rope2d<<<dim3((unsigned)n_tok, H), (unsigned)rope_dim>>>((bf16 *)d_q, (bf16 *)d_k, H, head_dim,
                                                                   (const int2 *)d_pos, rope_dim,
                                                                   (float)PULSAR_QWEN_VISION_ROPE_THETA);
        CUDA_LAUNCH("qwen vision rope2d");
        if (!vattention(d_q, d_k, d_v, d_attn, n_tok, H, head_dim, (float *)d_s, (bf16 *)d_p, (float *)d_o, s_entries))
            goto done;
        LINEAR(d_attn, b->proj_w, b->proj_b, d_tmp, D, D, n_tok);
        vk_add<<<(unsigned)((x_elems + VK_THREADS - 1) / VK_THREADS), VK_THREADS>>>((bf16 *)d_x, (const bf16 *)d_tmp,
                                                                                  x_elems);
        CUDA_LAUNCH("qwen vision attn residual");
        vk_layernorm<<<(unsigned)n_tok, ln_threads, ln_smem>>>((const bf16 *)d_x, WP(b->norm2_w), WP(b->norm2_b),
                                                              (bf16 *)d_tmp, D, eps);
        CUDA_LAUNCH("qwen vision norm2");
        LINEAR(d_tmp, b->fc1_w, b->fc1_b, d_mlp, D, I, n_tok);
        vk_gelu_tanh<<<(unsigned)(((size_t)n_tok * I + VK_THREADS - 1) / VK_THREADS), VK_THREADS>>>(
            (bf16 *)d_mlp, (size_t)n_tok * I);
        CUDA_LAUNCH("qwen vision gelu_tanh");
        LINEAR(d_mlp, b->fc2_w, b->fc2_b, d_attn, I, D, n_tok);
        vk_add<<<(unsigned)((x_elems + VK_THREADS - 1) / VK_THREADS), VK_THREADS>>>((bf16 *)d_x,
                                                                                  (const bf16 *)d_attn, x_elems);
        CUDA_LAUNCH("qwen vision mlp residual");
        if (li == 0) DUMP(1, d_x);
    }
    DUMP(2, d_x);

    /* merger: LayerNorm per patch, then each 2x2 block's 4 patches (consecutive: merge-block order) are one row */
    vk_layernorm<<<(unsigned)n_tok, ln_threads, ln_smem>>>((const bf16 *)d_x, WP(w->merger_norm_w),
                                                          WP(w->merger_norm_b), (bf16 *)d_tmp, D, eps);
    CUDA_LAUNCH("qwen vision merger norm");
    LINEAR(d_tmp, w->merger_fc1_w, w->merger_fc1_b, d_mid, M, M, n_out);
    vk_gelu<<<(unsigned)(((size_t)n_out * M + VK_THREADS - 1) / VK_THREADS), VK_THREADS>>>((bf16 *)d_mid,
                                                                                         (size_t)n_out * M);
    CUDA_LAUNCH("qwen vision merger gelu");
    LINEAR(d_mid, w->merger_fc2_w, w->merger_fc2_b, d_out, M, O, n_out);
    if ((size_t)n_out * O > (size_t)out_cap) goto done;
    if (cudaMemcpy(out_host, d_out, (size_t)n_out * O * sizeof(bf16), cudaMemcpyDeviceToHost) != cudaSuccess)
        goto done;
    ok = 1;
done:
    cudaFree(d_patches); cudaFree(d_x); cudaFree(d_tmp); cudaFree(d_attn); cudaFree(d_qkv); cudaFree(d_q);
    cudaFree(d_k); cudaFree(d_v); cudaFree(d_mlp); cudaFree(d_pos); cudaFree(d_idx); cudaFree(d_wt);
    cudaFree(d_mid); cudaFree(d_out); cudaFree(d_f32); cudaFree(d_s); cudaFree(d_p); cudaFree(d_o);
    return ok;
#undef DUMP
#undef LINEAR
#undef CUDA_LAUNCH
#undef CUDA_UP
#undef CUDA_ALLOC
#undef WP
}

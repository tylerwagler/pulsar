/* Engram device path (L242, V4.1): the gathered table rows become the `wkv` GEMM's
 * activation AS THE E4M3 + E8M0 THEY ALREADY ARE, and the gate writes the value into
 * every hyper-connection copy.
 *
 * Two kernels, both producer-side in the A8 sense:
 *
 *   engram_rows_emit_kernel   rows [n_tok][24][264 B] (256 E4M3 values, then the 8 E8M0
 *                             scales of their 32-blocks -- the row file's record, which is
 *                             the checkpoint's own bytes) -> the MXFP8 activation slot of a
 *                             [n_tok][6144] activation: data row-major, scales through
 *                             pulsar_mx_sfoff.  24 columns x 8 blocks = 192 k-blocks per token,
 *                             exactly what the slot's quantiser would have produced from a
 *                             dequantised row IF the dequantised value's block amax landed on
 *                             the stored scale -- which is why this is a copy and not a
 *                             quantise: the source's bytes are the value (rule 3).  The f32
 *                             plane of that activation is never written; the slot is noted
 *                             f32-skipped so no arm can read it.
 *
 *   engram_gate_add_kernel    per (token, hc copy): the reference's gate in f32 --
 *                             rstd = rsqrt(mean(h^2)+eps) * rsqrt(mean(key^2)+eps),
 *                             dot = sum(h * (q*k) * key) * rstd * dim^-0.5,
 *                             gate = sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot)),
 *                             h += gate * value -- on the bf16 residual copies (loaded to
 *                             f32, stored back rounded once, as `(h + ...).to(x.dtype)`).
 *                             A dead token (image span) keeps its copies untouched.
 *
 * The `wkv` GEMM between them is the ordinary MXFP8 entry on a weight registered by
 * offset like every pre-stored tensor. */
#include "pulsar_cuda_internal.h"
#include "pulsar_cuda_mx.cuh"

#define ENGRAM_DIM       256u
#define ENGRAM_N_SCALE   8u
#define ENGRAM_ROW_BYTES (ENGRAM_DIM + ENGRAM_N_SCALE)
#define ENGRAM_N_COLS    24u
#define ENGRAM_IN        (ENGRAM_N_COLS * ENGRAM_DIM)   /* 6144 */

/* one thread per (token, column, 32-block): 32 data bytes and one scale byte */
__global__ static void engram_rows_emit_kernel(const unsigned char *rows, uint32_t n_tok, int KBp,
                                               __nv_fp8_e4m3 *data, unsigned char *scale) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t per_tok = ENGRAM_N_COLS * ENGRAM_N_SCALE;   /* 192 blocks */
    if (i >= n_tok * per_tok) return;
    const uint32_t tok = i / per_tok, kb = i % per_tok;         /* kb = col*8 + j */
    const uint32_t col = kb / ENGRAM_N_SCALE, j = kb % ENGRAM_N_SCALE;
    /* A 264-byte record is 8-byte aligned, never 16: the loads are uint2.  The
     * destination block sits at a 32-byte boundary of the slot's data plane. */
    const unsigned char *row = rows + ((uint64_t)tok * ENGRAM_N_COLS + col) * ENGRAM_ROW_BYTES;
    const uint2 *src = (const uint2 *)(row + j * 32u);
    uint2 *dst = (uint2 *)(data + (uint64_t)tok * ENGRAM_IN + (uint64_t)kb * 32u);
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
    dst[3] = src[3];
    scale[pulsar_mx_sfoff((int)tok, (int)kb, KBp)] = row[ENGRAM_DIM + j];
}

int pulsar_gpu_engram_rows_emit(const pulsar_gpu_tensor *x_key, const pulsar_gpu_tensor *rows,
                                uint32_t n_tok) {
    if (!x_key || !rows || n_tok == 0) return 0;
    if (rows->bytes < (uint64_t)n_tok * ENGRAM_N_COLS * ENGRAM_ROW_BYTES ||
        x_key->bytes < (uint64_t)n_tok * ENGRAM_IN * sizeof(float)) {
        fprintf(stderr, "pulsar: engram rows emit: %u tokens need %llu row bytes and a %llu-byte key, "
                        "have %llu / %llu -- refusing\n", n_tok,
                (unsigned long long)n_tok * ENGRAM_N_COLS * ENGRAM_ROW_BYTES,
                (unsigned long long)n_tok * ENGRAM_IN * sizeof(float),
                (unsigned long long)rows->bytes, (unsigned long long)x_key->bytes);
        return 0;
    }
    if (((uintptr_t)rows->ptr & 7u) != 0) {
        fprintf(stderr, "pulsar: engram rows emit: the row block is not 8-byte aligned -- refusing\n");
        return 0;
    }
    void *dq = NULL, *dsf = NULL;
    int kbp = 0;
    if (!pulsar_gpu_mxfp8_act_cache_e4m3_slot(x_key, n_tok, ENGRAM_IN, &dq, &dsf, &kbp)) {
        fprintf(stderr, "pulsar: engram rows emit: no E4M3 slot for %u x %u -- refusing\n", n_tok, ENGRAM_IN);
        return 0;
    }
    const uint32_t n = n_tok * ENGRAM_N_COLS * ENGRAM_N_SCALE;
    engram_rows_emit_kernel<<<(n + 255u) / 256u, 256>>>((const unsigned char *)rows->ptr, n_tok, kbp,
                                                        (__nv_fp8_e4m3 *)dq, (unsigned char *)dsf);
    if (!cuda_ok(cudaGetLastError(), "engram rows emit")) return 0;
    pulsar_gpu_mxfp8_act_cache_arm(x_key, n_tok, ENGRAM_IN);
    pulsar_gpu_mxfp8_act_cache_note_mxfp8();
    pulsar_gpu_mxfp8_act_cache_note_f32_skipped(n_tok);   /* the f32 plane was never written */
    return 1;
}

/* one block per (token, hc copy); the three reductions in one pass, then the update */
__global__ static void engram_gate_add_kernel(pulsar_hc_t *hc, const float *kv, const float *w,
                                              const unsigned char *dead, uint32_t n_hc, uint32_t dim,
                                              float eps, float clamp) {
    const uint32_t tok = blockIdx.x / n_hc, c = blockIdx.x % n_hc;
    if (dead && dead[tok]) return;
    const uint64_t hbase = ((uint64_t)tok * n_hc + c) * dim;
    const float *key = kv + (uint64_t)tok * (n_hc + 1u) * dim + (uint64_t)c * dim;
    const float *value = kv + (uint64_t)tok * (n_hc + 1u) * dim + (uint64_t)n_hc * dim;
    const float *wc = w + (uint64_t)c * dim;
    float sh = 0.f, sk = 0.f, sd = 0.f;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        const float h = pulsar_hc_load(hc, hbase + i), k = key[i];
        sh += h * h;
        sk += k * k;
        sd += h * wc[i] * k;
    }
    __shared__ float red[3][32];
    sh = warp_sum_f32(sh); sk = warp_sum_f32(sk); sd = warp_sum_f32(sd);
    const uint32_t lane = threadIdx.x & 31u, wid = threadIdx.x >> 5;
    if (lane == 0) { red[0][wid] = sh; red[1][wid] = sk; red[2][wid] = sd; }
    __syncthreads();
    const uint32_t nw = (blockDim.x + 31u) >> 5;
    if (wid == 0) {
        sh = lane < nw ? red[0][lane] : 0.f;
        sk = lane < nw ? red[1][lane] : 0.f;
        sd = lane < nw ? red[2][lane] : 0.f;
        sh = warp_sum_f32(sh); sk = warp_sum_f32(sk); sd = warp_sum_f32(sd);
        if (lane == 0) { red[0][0] = sh; red[1][0] = sk; red[2][0] = sd; }
    }
    __syncthreads();
    const float inv_dim = 1.0f / (float)dim;
    const float rstd = rsqrtf(red[0][0] * inv_dim + eps) * rsqrtf(red[1][0] * inv_dim + eps);
    const float dot = red[2][0] * rstd * rsqrtf((float)dim);
    const float mag = sqrtf(fmaxf(fabsf(dot), clamp));
    const float gate = 1.0f / (1.0f + expf(-copysignf(mag, dot)));
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        const float h = pulsar_hc_load(hc, hbase + i);
        pulsar_hc_store(hc, hbase + i, h + gate * value[i]);
    }
}

int pulsar_gpu_engram_gate_add(pulsar_gpu_tensor *hc, const pulsar_gpu_tensor *kv,
                               const pulsar_gpu_tensor *qk_weight, const pulsar_gpu_tensor *dead,
                               uint32_t n_tok, uint32_t n_hc, uint32_t dim, float eps) {
    if (!hc || !kv || !qk_weight || n_tok == 0 || n_hc == 0 || dim == 0 || dim % 32u != 0) return 0;
    if (hc->bytes < (uint64_t)n_tok * n_hc * dim * PULSAR_HC_ELT_SIZE ||
        kv->bytes < (uint64_t)n_tok * (n_hc + 1u) * dim * sizeof(float) ||
        qk_weight->bytes < (uint64_t)n_hc * dim * sizeof(float) ||
        (dead && dead->bytes < n_tok)) {
        fprintf(stderr, "pulsar: engram gate: buffers too small for %u tokens x %u copies x %u -- refusing\n",
                n_tok, n_hc, dim);
        return 0;
    }
    engram_gate_add_kernel<<<n_tok * n_hc, 256>>>((pulsar_hc_t *)hc->ptr, (const float *)kv->ptr,
                                                  (const float *)qk_weight->ptr,
                                                  dead ? (const unsigned char *)dead->ptr : NULL,
                                                  n_hc, dim, eps, 1.0e-6f);
    return cuda_ok(cudaGetLastError(), "engram gate add");
}

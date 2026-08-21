#include "pulsar_cuda_internal.h"

/* Every KV buffer is PULSAR_ATTN_PACK rows -- 584 B at head_dim 512, nope dims
 * E4M3 with a per-64 E8M0 scale and rope dims bf16 -- so ONE decoder serves the
 * sliding-window ring, the compressed pool, the drafter's ring and the current
 * chunk alike.
 *
 * There WAS a per-call format flag here (`raw_f16`, later `raw_pack`). It died
 * on 2026-08-17 once the last f32 operand went: prefill's batch_kv, which
 * attention read directly while every other reader took packed rows. I tried to
 * delete this flag three times before that and was wrong each time -- the
 * drafter's ring, then prefill's batch_kv -- so it is worth saying plainly that
 * what made it removable was removing its last CALLER, not re-reading the
 * grep. */
__device__ static inline float raw_kv_ld(const float *raw_kv, uint64_t row,
                                         uint32_t d, uint32_t head_dim) {
    return attn_comp_pack_ld(raw_kv, row, d, head_dim);
}

/* The c4-th group of four dims of `row`; packed rows are byte-addressed with a
 * per-64 scale, so the four come scalar. */
__device__ static inline float4 raw_kv_ld4(const float *raw_kv, uint64_t row,
                                           uint32_t c4, uint32_t head_dim) {
    const uint32_t d0 = c4 << 2;
    float4 v;
    v.x = attn_comp_pack_ld(raw_kv, row, d0 + 0u, head_dim);
    v.y = attn_comp_pack_ld(raw_kv, row, d0 + 1u, head_dim);
    v.z = attn_comp_pack_ld(raw_kv, row, d0 + 2u, head_dim);
    v.w = attn_comp_pack_ld(raw_kv, row, d0 + 3u, head_dim);
    return v;
}

/* THE COMP CACHE OPERAND IS PULSAR_ATTN_PACK ROWS.  Always -- there is no
 * format parameter any more.
 *
 * A per-call comp_kv_pack flag chose between packed and f32 comp rows here,
 * threaded through every kernel and seam entry.  Its f32 arm became unreachable
 * when the last f32 shadow pool went (2026-08-17); it was removed 2026-08-18.
 * Worth recording why such a flag is not free: both formats are `const float *`,
 * so passing it wrong reads 584 B rows at a 2048 B stride -- out of bounds, NaN,
 * and a clean compile.  That already happened once, on the f16 prefill entry
 * that hard-coded 0.
 *
 * The row is
 * (see PULSAR_ATTN_PACK_* in pulsar_cuda_internal.h): [n_nope e4m3][n_nope/64 E8M0]
 * [pad][n_rot bf16 rope].  Nope dims decode e4m3_value * 2^(e8-127) — exactly
 * the fp8_kv_quantize roundtrip value the f32 cache holds — and rope dims widen
 * from bf16, which is likewise exactly what the f32 cache holds since the pack
 * store roundtrips the rope tail in place, so scores/outputs are bit-identical
 * to the f32 comp cache.
 * The 2^(e8-127) scale is built as a float exponent; the pack amax floor
 * (1e-4 -> e8 >= 105) rules out byte 0. */
/* attn_pack_e4m3 / attn_comp_pack_ld now live in pulsar_cuda_internal.h so the
 * fp16 tensor-core kernel decodes ATTN_PACK rows with the SAME code rather
 * than a transcription of it -- there is then no second copy of the contract
 * to drift or to get wrong. */

/* Packed-row dot walked d = 0..head_dim-1 by one thread: per-64-block scale
 * hoisted, e4m3 bytes fetched four at a time as one uint32 (rows are 4-byte
 * aligned: 584-byte stride).  Accumulation order is identical to the scalar
 * d-ascending loop, so the result is bit-identical. */
template <typename QT>
__device__ static inline float attn_pack_dot_full(const QT *qh, const float *comp_kv, uint64_t row, uint32_t head_dim, float dot) {
    const uint32_t n_nope = head_dim - PULSAR_ATTN_PACK_NROT;
    const uint8_t *pr = (const uint8_t *)comp_kv + row * PULSAR_ATTN_PACK_ROWBYTES(head_dim);
    const uint8_t *psc = pr + n_nope;
    for (uint32_t off = 0; off < n_nope; off += PULSAR_FP8_KV_BLOCK) {
        const float scale = __uint_as_float((uint32_t)psc[off / PULSAR_FP8_KV_BLOCK] << 23);
        const uint32_t *pw = (const uint32_t *)(pr + off);
        for (uint32_t i = 0; i < PULSAR_FP8_KV_BLOCK / 4u; i++) {
            const uint32_t w = pw[i];
            const uint32_t d = off + i * 4u;
            dot += q_load<QT>(qh, d + 0u) * attn_pack_e4m3(w & 0xffu, scale);
            dot += q_load<QT>(qh, d + 1u) * attn_pack_e4m3((w >> 8) & 0xffu, scale);
            dot += q_load<QT>(qh, d + 2u) * attn_pack_e4m3((w >> 16) & 0xffu, scale);
            dot += q_load<QT>(qh, d + 3u) * attn_pack_e4m3(w >> 24, scale);
        }
    }
    const __nv_bfloat16 *rope = (const __nv_bfloat16 *)(pr + n_nope + PULSAR_ATTN_PACK_SCALES_PAD(head_dim));
    for (uint32_t d = 0; d < PULSAR_ATTN_PACK_NROT; d++) dot += q_load<QT>(qh, n_nope + d) * __bfloat162float(rope[d]);
    return dot;
}

/* Packed-row dot walked d = lane, lane+8, ... by one thread (8-lane strided
 * kernels): per-64-block scale hoisted (8 dims per block per thread share it).
 * Same d-ascending visit order as the plain strided loop — bit-identical. */
template <typename QT>
__device__ static inline float attn_pack_dot_lane8(const QT *qh, const float *comp_kv, uint64_t row, uint32_t lane, uint32_t head_dim, float dot) {
    const uint32_t n_nope = head_dim - PULSAR_ATTN_PACK_NROT;
    const uint8_t *pr = (const uint8_t *)comp_kv + row * PULSAR_ATTN_PACK_ROWBYTES(head_dim);
    const uint8_t *psc = pr + n_nope;
    for (uint32_t off = 0; off < n_nope; off += PULSAR_FP8_KV_BLOCK) {
        const float scale = __uint_as_float((uint32_t)psc[off / PULSAR_FP8_KV_BLOCK] << 23);
        for (uint32_t d = off + lane; d < off + PULSAR_FP8_KV_BLOCK; d += 8u) {
            dot += q_load<QT>(qh, d) * attn_pack_e4m3(pr[d], scale);
        }
    }
    const __nv_bfloat16 *rope = (const __nv_bfloat16 *)(pr + n_nope + PULSAR_ATTN_PACK_SCALES_PAD(head_dim));
    for (uint32_t d = n_nope + lane; d < head_dim; d += 8u) dot += q_load<QT>(qh, d) * __bfloat162float(rope[d - n_nope]);
    return dot;
}

template <typename QT>
__global__ static void attention_prefill_raw_kernel(
        float *heads,
        const float *sinks,
        const QT *q,
        const float *raw_kv,
        uint32_t n_tokens,
        uint32_t window,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    /* window==0 means unlimited, as the mixed/softmax/decode kernels all treat
     * it (:172, :251, :309, :450) -- their convention is "all rows k<=t
     * visible". This kernel's old `min(t+1, window)` instead yielded 0 rows at
     * window==0, disagreeing with its own softmax normalizer (:251) and
     * corrupting the row if a full-attention config ever reached the raw tier.
     * For window>0 this is bit-identical to min(t+1, window). */
    uint32_t raw_count = (window != 0u && window < t + 1u) ? window : t + 1u;
    uint32_t raw_start = t + 1 - raw_count;
    const QT *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    __shared__ float scores[256];
    __shared__ float partial[128];
    __shared__ float max_s;
    __shared__ float denom;
    float scale = rsqrtf((float)head_dim);
    float local_max = sinks[h];
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) dot += q_load<QT>(qh, d) * raw_kv_ld(raw_kv, (uint64_t)(raw_start + r), d, head_dim);
        scores[r] = dot * scale;
        local_max = fmaxf(local_max, scores[r]);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    if (threadIdx.x == 0) {
        float den = expf(sinks[h] - max_s);
        for (uint32_t r = 0; r < raw_count; r++) {
            scores[r] = expf(scores[r] - max_s);
            den += scores[r];
        }
        denom = den;
    }
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            acc += raw_kv_ld(raw_kv, (uint64_t)(raw_start + r), d, head_dim) * scores[r];
        }
        oh[d] = acc / denom;
    }
}

template <typename QT>
__global__ static void attention_prefill_mixed_kernel(
        float *heads,
        const float *sinks,
        const QT *q,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    const QT *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    uint32_t raw_start = (window != 0 && t + 1u > window) ? t + 1u - window : 0u;
    uint32_t raw_count = t + 1u - raw_start;
    uint32_t visible_comp = (t + 1u) / ratio;
    if (visible_comp > n_comp) visible_comp = n_comp;
    __shared__ float scores[512];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float scale = rsqrtf((float)head_dim);
    float local_max = sinks[h];
    uint32_t n_score = raw_count + visible_comp;

    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) dot += q_load<QT>(qh, d) * raw_kv_ld(raw_kv, (uint64_t)(raw_start + r), d, head_dim);
        scores[r] = dot * scale;
        local_max = fmaxf(local_max, scores[r]);
    }
    for (uint32_t c = threadIdx.x; c < visible_comp; c += blockDim.x) {
        float s = -INFINITY;
        {
            /* Packed rows go through attn_pack_dot_full -- the same walk the
              * decode kernels use, so there is one decoder for the one format.
              * This kernel could ONLY read f32 until 2026-08-17, which is why a
              * whole f32 shadow of the packed pool existed to feed it. */
            float dot = 0.0f;
            dot = attn_pack_dot_full(qh, comp_kv, (uint64_t)c, head_dim, 0.0f);

            s = dot * scale;
        }
        scores[raw_count + c] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv_ld(raw_kv, (uint64_t)(raw_start + r), d, head_dim) * scores[r];
        for (uint32_t c = 0; c < visible_comp; c++)
            acc += (attn_comp_pack_ld(comp_kv, (uint64_t)c, d, head_dim)) * scores[raw_count + c];
        oh[d] = acc / denom;
    }
}

__global__ static void attention_prefill_raw_softmax_kernel(
        float *scores,
        const float *sinks,
        uint32_t n_tokens,
        uint32_t window,
        uint32_t n_keys) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens) return;
    float *row = scores + ((uint64_t)h * n_tokens + t) * n_keys;
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float local_max = sinks[h];
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        bool valid = k <= t && (window == 0 || t - k < window);
        float s = valid ? row[k] : -INFINITY;
        row[k] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float p = isfinite(row[k]) ? expf(row[k] - max_s) : 0.0f;
        row[k] = p;
        den_local += p;
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) row[k] /= denom;
}

__global__ static void attention_prefill_mixed_softmax_kernel(
        float *scores,
        const float *sinks,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_keys,
        /* When non-NULL the normalized probabilities are stored HERE as fp16
         * and the f32 store is skipped: the PV GEMM reads only one of the two,
         * so this is one store instead of two, and it removes the separate
         * f32->fp16 pass over the whole score matrix (which was memory-bound
         * enough to eat the entire tensor-core win, measured 2026-08-05). */
        __half *scores_h) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || ratio == 0) return;
    const uint64_t row_off = ((uint64_t)h * n_tokens + t) * n_keys;
    float *row = scores + row_off;
    __half *row_h = scores_h ? scores_h + row_off : NULL;
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float local_max = sinks[h];
    const uint32_t visible_comp = (t + 1u) / ratio;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float s = -INFINITY;
        if (k < n_tokens) {
            if (k <= t && (window == 0 || t - k < window)) s = row[k];
        } else {
            uint32_t c = k - n_tokens;
            if (c < n_comp && c < visible_comp) {
                s = row[k];
            }
        }
        row[k] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float p = isfinite(row[k]) ? expf(row[k] - max_s) : 0.0f;
        row[k] = p;
        den_local += p;
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        const float p = row[k] / denom;
        if (row_h) row_h[k] = __float2half(p);
        else row[k] = p;
    }
}

__global__ static void attention_prefill_pack_mixed_kv_kernel(
        float *dst,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t head_dim,
        /* Same idea as the softmax fp16 store: emit the packed KV directly in
         * the GEMM's operand type instead of writing f32 and converting after. */
        __half *dst_h) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)(n_tokens + n_comp) * head_dim;
    if (gid >= n) return;
    uint32_t d = gid % head_dim;
    uint32_t r = gid / head_dim;
    const float v = r < n_tokens
            ? raw_kv_ld(raw_kv, (uint64_t)r, d, head_dim)
            : attn_comp_pack_ld(comp_kv, (uint64_t)(r - n_tokens), d, head_dim);
    if (dst_h) dst_h[gid] = __float2half(v);
    else dst[gid] = v;
}

__global__ static void attention_prefill_unpack_heads_kernel(
        float *heads,
        const float *tmp,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t head_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
    if (gid >= n) return;
    uint32_t d = gid % head_dim;
    uint64_t q = gid / head_dim;
    uint32_t h = q % n_head;
    uint32_t t = q / n_head;
    heads[gid] = tmp[((uint64_t)h * n_tokens + t) * head_dim + d];
}

/* positions/seq_id/comp_cap (all descriptor-aware decode kernels): per-row
 * multi-session banking (design adapted from the MIT-licensed Entrpi/ds4
 * fork, v0.2 c71a49a — see pulsar_bank_slabs in the engine; reimplemented, no
 * code copied).  positions[t] is row t's absolute query position, seq_id[t]
 * its TRUE bank id (never a packed row ordinal); raw window, ring start and
 * the visible compressed count are derived per row from the position because
 * the raw ring is position-indexed (slot = pos % raw_cap per bank) and
 * compression closes one ratio group every `ratio` positions.  Banked rows
 * read the raw ring at seq_id*raw_cap and compressed rows at seq_id*comp_cap
 * offsets; the scalar n_comp becomes a cross-bank superset used ONLY as a
 * clamp/scratch bound, never to address into a specific bank.  The per-row
 * visible count is (qpos+1)/ratio — the SAME rule the engine's classic
 * single-session decode follows, because the engine emits a step's
 * compressed row BEFORE attention (gpu_decode.cpp: layer_n_comp is
 * incremented before the attention launch reads it), so at an emit step
 * (qpos ≡ ratio-1 mod ratio) the row attends to the compressed row emitted
 * that same step.  DRIVER CONTRACT (banked mode): every bank's compressed
 * rows for the current step — including same-step emits — must be written
 * before the attention launch; the scalar n_comp superset clamp is a safety
 * bound only.  If the clamp ever bites, the row reads fewer rows than
 * single-session would (fail-safe, not garbage) and its output DIVERGES
 * from classic — that is the mid-prefill-bank case the driver must never
 * co-schedule.  positions == NULL && seq_id == NULL degenerates to the
 * classic single-cache scalar path bit-exactly. */
template <typename QT>
__global__ static void attention_decode_mixed_kernel(
        float *heads,
        const float *sinks,
        const QT *q,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t non_causal,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,
        const int32_t * __restrict__ positions,
        const int32_t * __restrict__ seq_id,
        const void * const * __restrict__ comp_bank_ptrs,
        uint32_t comp_cap,
        uint32_t n_banks) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    if (seq_id && (uint32_t)seq_id[t] >= n_banks) {
        /* Dead/evicted row (stale or sentinel bank id — the host cannot audit
         * a device array): fail-visible.  Zero this row's head output and
         * read nothing; an out-of-pool id would otherwise be a silent wild
         * read across the whole slab. */
        float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) oh[d] = 0.0f;
        return;
    }
    const bool single_all = (n_tokens == 1u && ratio == 0u && positions == NULL);
    uint32_t qpos = positions ? (uint32_t)positions[t] : pos0 + t;
    uint32_t eff_n_raw = n_raw;
    uint32_t eff_raw_start = raw_start;
    uint32_t first_raw_pos;
    if (positions) {
        eff_n_raw = (window != 0u && qpos + 1u > window) ? window : qpos + 1u;
        if (eff_n_raw > raw_cap) eff_n_raw = raw_cap;
        eff_raw_start = (qpos + 1u - eff_n_raw) % raw_cap;
        first_raw_pos = qpos + 1u - eff_n_raw;
    } else {
        first_raw_pos = pos0 + n_tokens - n_raw;
    }
    const uint32_t raw_base = seq_id ? (uint32_t)seq_id[t] * raw_cap : 0u;
    /* Per-bank comp base: with split allocations the batched path passes a base-
     * pointer table (comp_bank_ptrs), so bank sid's comp cache is a separate
     * allocation read at LOCAL row (comp_base == 0). NULL table → the classic
     * single-slab base + seq_id*comp_cap addressing, byte-identical. */
    const uint32_t sid_b = seq_id ? (uint32_t)seq_id[t] : 0u;
    const float *comp_src = comp_bank_ptrs ? (const float *)comp_bank_ptrs[sid_b] : comp_kv;
    const uint64_t comp_base = comp_bank_ptrs ? 0u
                             : (seq_id ? (uint64_t)sid_b * comp_cap : 0u);
    uint32_t visible_comp = single_all ? n_comp : (n_comp ? (qpos + 1u) / ratio : 0u);
    if (visible_comp > n_comp) visible_comp = n_comp;
    const QT *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    __shared__ float scores[PULSAR_CUDA_ATTENTION_SCORE_CAP];
    __shared__ uint32_t raw_rows[256];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    float scale = rsqrtf((float)head_dim);
    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        if (eff_n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + eff_n_raw - 1u;
            if (single_all) {
                raw_count = eff_n_raw > 256u ? 256u : eff_n_raw;
            } else if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = non_causal ? raw_last_pos : (qpos < raw_last_pos ? qpos : raw_last_pos);
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = raw_base + (eff_raw_start + raw_first_idx + r) % raw_cap;
    }
    __syncthreads();
    uint32_t n_score = raw_count + visible_comp;
    float local_max = sinks[h];
    if (visible_comp == 0 || n_tokens == 1u) {
        for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
            const uint64_t rrow = (uint64_t)raw_rows[r];
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += q_load<QT>(qh, d) * raw_kv_ld(raw_kv, rrow, d, head_dim);
            scores[r] = dot * scale;
            local_max = fmaxf(local_max, scores[r]);
        }
        for (uint32_t c = threadIdx.x; c < visible_comp; c += blockDim.x) {
            float s = -INFINITY;
            {
                float dot = 0.0f;
                dot = attn_pack_dot_full(qh, comp_src, comp_base + c, head_dim, dot);

                s = dot * scale;
            }
            scores[raw_count + c] = s;
            local_max = fmaxf(local_max, s);
        }
    } else {
        uint32_t qlane = threadIdx.x & 7u;
        uint32_t qgroup = threadIdx.x >> 3u;
        for (uint32_t row0 = 0; row0 < n_score; row0 += 32u) {
            uint32_t row = row0 + qgroup;
            if (row < n_score) {
                bool have_row = true;
                uint32_t c_idx = row < raw_count ? 0u : row - raw_count;
                float s = -INFINITY;
                if (have_row) {
                    float dot = 0.0f;
                    if (row < raw_count) {
                        const uint64_t rrow = (uint64_t)raw_rows[row];
                        for (uint32_t d = qlane; d < head_dim; d += 8u) dot += q_load<QT>(qh, d) * raw_kv_ld(raw_kv, rrow, d, head_dim);
                    } else {
                        dot = attn_pack_dot_lane8(qh, comp_src, comp_base + c_idx, qlane, head_dim, dot);
                    }
                    const uint32_t mask = 0xffu << (threadIdx.x & 24u);
                    for (uint32_t off = 4u; off > 0u; off >>= 1u) {
                        dot += __shfl_down_sync(mask, dot, off, 8);
                    }
                    s = dot * scale;
                }
                if (qlane == 0) scores[row] = s;
            }
        }
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
            local_max = fmaxf(local_max, scores[i]);
        }
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    if (head_dim == 512u && blockDim.x == 256u) {
        uint32_t d0 = threadIdx.x;
        uint32_t d1 = d0 + 256u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            float s = scores[r];
            const uint64_t rrow = (uint64_t)raw_rows[r];
            acc0 += raw_kv_ld(raw_kv, rrow, d0, head_dim) * s;
            acc1 += raw_kv_ld(raw_kv, rrow, d1, head_dim) * s;
        }
        for (uint32_t c = 0; c < visible_comp; c++) {
            float s = scores[raw_count + c];
            acc0 += attn_comp_pack_ld(comp_src, comp_base + c, d0, head_dim) * s;
            acc1 += attn_comp_pack_ld(comp_src, comp_base + c, d1, head_dim) * s;

        }
        oh[d0] = acc0 / denom;
        oh[d1] = acc1 / denom;
    } else {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv_ld(raw_kv, (uint64_t)raw_rows[r], d, head_dim) * scores[r];
            for (uint32_t c = 0; c < visible_comp; c++) {
                acc += attn_comp_pack_ld(comp_src, comp_base + c, d, head_dim) * scores[raw_count + c];

            }
            oh[d] = acc / denom;
        }
    }
}

template <typename QT>
__global__ static void attention_indexed_mixed_kernel(
        float *heads,
        const float *sinks,
        const QT *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,
        const int32_t * __restrict__ positions,
        const int32_t * __restrict__ seq_id,
        const void * const * __restrict__ comp_bank_ptrs,
        uint32_t comp_cap,
        uint32_t n_banks) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    if (seq_id && (uint32_t)seq_id[t] >= n_banks) {
        /* Dead/evicted row: see attention_decode_mixed_kernel. */
        float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) oh[d] = 0.0f;
        return;
    }
    /* Descriptor preamble: see attention_decode_mixed_kernel.  comp_rows[]
     * keeps bank-LOCAL compressed ids (top-k ids are per-bank); the bank
     * offset is applied at read time via comp_base. */
    uint32_t qpos = positions ? (uint32_t)positions[t] : pos0 + t;
    uint32_t eff_n_raw = n_raw;
    uint32_t eff_raw_start = raw_start;
    uint32_t first_raw_pos;
    if (positions) {
        eff_n_raw = (window != 0u && qpos + 1u > window) ? window : qpos + 1u;
        if (eff_n_raw > raw_cap) eff_n_raw = raw_cap;
        eff_raw_start = (qpos + 1u - eff_n_raw) % raw_cap;
        first_raw_pos = qpos + 1u - eff_n_raw;
    } else {
        first_raw_pos = pos0 + n_tokens - n_raw;
    }
    const uint32_t raw_base = seq_id ? (uint32_t)seq_id[t] * raw_cap : 0u;
    /* Per-bank comp base (see attention_decode_mixed_kernel): base-pointer table
     * → separate per-bank allocation at LOCAL row; NULL → classic seq_id*comp_cap. */
    const uint32_t sid_b = seq_id ? (uint32_t)seq_id[t] : 0u;
    const float *comp_src = comp_bank_ptrs ? (const float *)comp_bank_ptrs[sid_b] : comp_kv;
    const uint64_t comp_base = comp_bank_ptrs ? 0u
                             : (seq_id ? (uint64_t)sid_b * comp_cap : 0u);
    uint32_t visible_comp = n_comp;
    if (ratio != 0) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }
    const QT *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    __shared__ float scores[768];
    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t comp_rows[512];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    __shared__ uint32_t comp_count;
    float scale = rsqrtf((float)head_dim);
    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        comp_count = 0;
        if (eff_n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + eff_n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = raw_base + (eff_raw_start + raw_first_idx + r) % raw_cap;
    }
    /* Deterministic ordered compaction (was an atomicAdd slot race): the
     * ORDER of comp_rows fixes the float accumulation order of the indexed
     * attention that consumes it, so it must not depend on warp scheduling.
     * Order-preserving parallel compaction: parallel load + Hillis-Steele
     * scan over the validity flags in smem. top_k <= 512, and the add[2]
     * double-buffer needs blockDim.x >= lim/2 (launch is <<<grid, 256>>>). */
    __shared__ int32_t s_topk[512];
    __shared__ uint16_t s_scan[512];
    const uint32_t lim = top_k < 512u ? top_k : 512u;
    for (uint32_t i = threadIdx.x; i < lim; i += blockDim.x) {
        const int32_t c = topk[(uint64_t)t * top_k + i];
        s_topk[i] = c;
        s_scan[i] = (uint16_t)(c >= 0 && (uint32_t)c < visible_comp);
    }
    __syncthreads();
    for (uint32_t stride = 1; stride < lim; stride <<= 1) {
        uint16_t add[2] = {0, 0};
        for (uint32_t i = threadIdx.x, k = 0; i < lim; i += blockDim.x, k++)
            add[k & 1] = i >= stride ? s_scan[i - stride] : (uint16_t)0;
        __syncthreads();
        for (uint32_t i = threadIdx.x, k = 0; i < lim; i += blockDim.x, k++)
            s_scan[i] = (uint16_t)(s_scan[i] + add[k & 1]);
        __syncthreads();
    }
    for (uint32_t i = threadIdx.x; i < lim; i += blockDim.x) {
        const int32_t c = s_topk[i];
        if (c >= 0 && (uint32_t)c < visible_comp)
            comp_rows[s_scan[i] - 1u] = (uint32_t)c;   /* inclusive scan -> slot */
    }
    if (threadIdx.x == 0) comp_count = lim ? s_scan[lim - 1u] : 0u;
    __syncthreads();
    uint32_t n_score = raw_count + comp_count;
    float local_max = sinks[h];
    if (comp_count == 0) {
        for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
            const uint64_t rrow = (uint64_t)raw_rows[r];
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += q_load<QT>(qh, d) * raw_kv_ld(raw_kv, rrow, d, head_dim);
            scores[r] = dot * scale;
            local_max = fmaxf(local_max, scores[r]);
        }
    } else {
        uint32_t qlane = threadIdx.x & 7u;
        uint32_t qgroup = threadIdx.x >> 3u;
        for (uint32_t row0 = 0; row0 < n_score; row0 += 32u) {
            uint32_t row = row0 + qgroup;
            if (row < n_score) {
                float dot = 0.0f;
                if (row < raw_count) {
                    const uint64_t rrow = (uint64_t)raw_rows[row];
                    for (uint32_t d = qlane; d < head_dim; d += 8u) dot += q_load<QT>(qh, d) * raw_kv_ld(raw_kv, rrow, d, head_dim);
                } else {
                    dot = attn_pack_dot_lane8(qh, comp_src, comp_base + comp_rows[row - raw_count], qlane, head_dim, dot);
                }
                const uint32_t mask = 0xffu << (threadIdx.x & 24u);
                for (uint32_t off = 4u; off > 0u; off >>= 1u) {
                    dot += __shfl_down_sync(mask, dot, off, 8);
                }
                if (qlane == 0) scores[row] = dot * scale;
            }
        }
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
            local_max = fmaxf(local_max, scores[i]);
        }
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    if (head_dim == 512u && blockDim.x == 256u) {
        uint32_t d0 = threadIdx.x;
        uint32_t d1 = d0 + 256u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            float s = scores[r];
            const uint64_t rrow = (uint64_t)raw_rows[r];
            acc0 += raw_kv_ld(raw_kv, rrow, d0, head_dim) * s;
            acc1 += raw_kv_ld(raw_kv, rrow, d1, head_dim) * s;
        }
        for (uint32_t c = 0; c < comp_count; c++) {
            float s = scores[raw_count + c];
            acc0 += attn_comp_pack_ld(comp_src, comp_base + comp_rows[c], d0, head_dim) * s;
            acc1 += attn_comp_pack_ld(comp_src, comp_base + comp_rows[c], d1, head_dim) * s;

        }
        oh[d0] = acc0 / denom;
        oh[d1] = acc1 / denom;
    } else {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv_ld(raw_kv, (uint64_t)raw_rows[r], d, head_dim) * scores[r];
            for (uint32_t s = 0; s < comp_count; s++) {
                acc += (attn_comp_pack_ld(comp_src, comp_base + comp_rows[s], d, head_dim)) * scores[raw_count + s];
            }
            oh[d] = acc / denom;
        }
    }
}

/* Occupancy knob for the prefill attention kernel.  ncu at the shipped shape:
 * 70 registers/thread x 512 threads = 35840 regs/block against 65536 per SM, so
 * exactly ONE block fits and occupancy pins at 33.3% (Block Limit Registers = 1,
 * Block Limit Shared Mem = 1 -- both binding).  The dominant stall is
 * short_scoreboard (MIO / shared memory) at 3.49 of ~9.3 cycles per
 * issue-active, so more resident warps is the obvious lever.
 *
 * MEASURED, harness `tests/attn_indexed_bench.cu` at the shipped shape:
 *   min_blocks   sub-batch 4   sub-batch 7
 *   (none)          9.957         10.098 ms
 *   2               6.998          7.097 ms   <- 1.42x, shipped
 *   3              79.839         79.911 ms   <- register spill, 8x WORSE
 * In-engine: prefill @4k 670.84 -> 730.71 (+8.9%), @8k 647.19 -> 702.79.
 * Frontier logits are BYTE-IDENTICAL with and without, so this is free speed,
 * not a precision trade -- verified by gguf-tools/lb_numerics_check.sh.
 * Override with -DPULSAR_ATTN_MIN_BLOCKS=N; 3 is a cliff, do not raise blindly. */
#ifndef PULSAR_ATTN_MIN_BLOCKS
#define PULSAR_ATTN_MIN_BLOCKS 2
#endif
#define PULSAR_ATTN_LB __launch_bounds__(512, PULSAR_ATTN_MIN_BLOCKS)

template <uint32_t ROWS_PER_STAGE, uint32_t HEADS_PER_GROUP, typename QT>
__global__ PULSAR_ATTN_LB static void attention_indexed_mixed_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const QT *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,
        const int32_t * __restrict__ positions,
        const int32_t * __restrict__ seq_id,
        const void * const * __restrict__ comp_bank_ptrs,
        uint32_t comp_cap,
        uint32_t n_banks) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * HEADS_PER_GROUP + warp;
    const bool valid_head = head < n_head;
    if (seq_id && (uint32_t)seq_id[t] >= n_banks) {
        /* Dead/evicted row: zero the head, then all threads return together
         * (no __syncthreads has run yet).  See
         * attention_decode_mixed_heads8_online_kernel. */
        if (valid_head) {
            float *oh = heads + ((uint64_t)t * n_head + head) * head_dim;
            for (uint32_t d = lane; d < head_dim; d += 32u) oh[d] = 0.0f;
        }
        return;
    }

    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    __shared__ float4 kv_shared[ROWS_PER_STAGE * 128];

    /* Descriptor (banked) preamble: per-row qpos, raw ring span, per-bank base
     * derived from positions[t]/seq_id[t], byte-for-byte as
     * attention_decode_mixed_heads8_online_kernel derives them.  NULL
     * descriptors reproduce the classic scalar pos0+t / raw_start / contiguous
     * comp path exactly (raw_base = comp_base = 0, comp_src = comp_kv). */
    const uint32_t qpos = positions ? (uint32_t)positions[t] : pos0 + t;
    uint32_t eff_n_raw = n_raw;
    uint32_t eff_raw_start = raw_start;
    uint32_t first_raw_pos;
    if (positions) {
        eff_n_raw = (window != 0u && qpos + 1u > window) ? window : qpos + 1u;
        if (eff_n_raw > raw_cap) eff_n_raw = raw_cap;
        eff_raw_start = (qpos + 1u - eff_n_raw) % raw_cap;
        first_raw_pos = qpos + 1u - eff_n_raw;
    } else {
        first_raw_pos = pos0 + n_tokens - n_raw;
    }
    const uint32_t raw_base = seq_id ? (uint32_t)seq_id[t] * raw_cap : 0u;
    const uint32_t sid_b = seq_id ? (uint32_t)seq_id[t] : 0u;
    const float *comp_src = comp_bank_ptrs ? (const float *)comp_bank_ptrs[sid_b] : comp_kv;
    const uint64_t comp_base = comp_bank_ptrs ? 0u
                             : (seq_id ? (uint64_t)sid_b * comp_cap : 0u);
    uint32_t visible_comp = n_comp;
    if (ratio != 0) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }

    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        if (eff_n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + eff_n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = raw_base + (eff_raw_start + raw_first_idx + r) % raw_cap;
    }
    __syncthreads();

    uint32_t comp_count = top_k < visible_comp ? top_k : visible_comp;
    if (comp_count > 512u) comp_count = 512u;
    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const QT *qrow = valid_head
        ? (q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q_load4<QT>(qrow, lane +  0u);
        q1 = q_load4<QT>(qrow, lane + 32u);
        q2 = q_load4<QT>(qrow, lane + 64u);
        q3 = q_load4<QT>(qrow, lane + 96u);
    }

    float max_s = -INFINITY;
    float sum_s = 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t row0 = 0; row0 < n_score; row0 += ROWS_PER_STAGE) {
        const uint32_t nr = n_score - row0 < ROWS_PER_STAGE ? n_score - row0 : ROWS_PER_STAGE;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            /* Clamp the top-k index to visible_comp: the engine's invariant keeps
             * padding sentinels (UINT32_MAX) out of this path, but a stray value
             * would otherwise be a ~4 GB wild read. Substitute row 0 on violation. */
            uint32_t comp_idx = 0u;
            if (sr >= raw_count) {
                int32_t c = topk[(uint64_t)t * top_k + (sr - raw_count)];
                comp_idx = (c >= 0 && (uint32_t)c < visible_comp) ? (uint32_t)c : 0u;
            }
            if (sr < raw_count) {
                kv_shared[off] = raw_kv_ld4(raw_kv, (uint64_t)raw_rows[sr], c4, head_dim);
            } else {
                /* ATTN_PACK comp row -> f32 float4 in smem, byte-identical to the
                 * pack dequant in attention_decode_mixed_heads8_online_kernel; the
                 * indexed row id is the top-k comp_idx, not sr-raw_count. */
                const uint32_t n_nope = head_dim - PULSAR_ATTN_PACK_NROT;
                const uint8_t *pr = (const uint8_t *)comp_src +
                    (comp_base + (uint64_t)comp_idx) * PULSAR_ATTN_PACK_ROWBYTES(head_dim);
                const uint32_t base = c4 << 2;
                float4 v;
                if (base < n_nope) {
                    const float scale = __uint_as_float((uint32_t)pr[n_nope + (base / PULSAR_FP8_KV_BLOCK)] << 23);
                    const uint32_t w = *(const uint32_t *)(pr + base);
                    v.x = attn_pack_e4m3(w & 0xffu, scale);
                    v.y = attn_pack_e4m3((w >> 8) & 0xffu, scale);
                    v.z = attn_pack_e4m3((w >> 16) & 0xffu, scale);
                    v.w = attn_pack_e4m3(w >> 24, scale);
                } else {
                    const __nv_bfloat16 *rope = (const __nv_bfloat16 *)(pr + n_nope + PULSAR_ATTN_PACK_SCALES_PAD(head_dim));
                    v.x = __bfloat162float(rope[base - n_nope + 0u]);
                    v.y = __bfloat162float(rope[base - n_nope + 1u]);
                    v.z = __bfloat162float(rope[base - n_nope + 2u]);
                    v.w = __bfloat162float(rope[base - n_nope + 3u]);
                }
                kv_shared[off] = v;
            }
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) +
                              dot4_f32(q1, k1) +
                              dot4_f32(q2, k2) +
                              dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(0xffffffffu, score, 0);

                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m);
                const float row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + k0.x * row_scale;
                o0.y = o0.y * old_scale + k0.y * row_scale;
                o0.z = o0.z * old_scale + k0.z * row_scale;
                o0.w = o0.w * old_scale + k0.w * row_scale;
                o1.x = o1.x * old_scale + k1.x * row_scale;
                o1.y = o1.y * old_scale + k1.y * row_scale;
                o1.z = o1.z * old_scale + k1.z * row_scale;
                o1.w = o1.w * old_scale + k1.w * row_scale;
                o2.x = o2.x * old_scale + k2.x * row_scale;
                o2.y = o2.y * old_scale + k2.y * row_scale;
                o2.z = o2.z * old_scale + k2.z * row_scale;
                o2.w = o2.w * old_scale + k2.w * row_scale;
                o3.x = o3.x * old_scale + k3.x * row_scale;
                o3.y = o3.y * old_scale + k3.y * row_scale;
                o3.z = o3.z * old_scale + k3.z * row_scale;
                o3.w = o3.w * old_scale + k3.w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }

    if (valid_head) {
        const float sink = sinks[head];
        const float new_m = fmaxf(max_s, sink);
        const float old_scale = expf(max_s - new_m);
        const float sink_scale = expf(sink - new_m);
        sum_s = sum_s * old_scale + sink_scale;
        o0.x *= old_scale; o0.y *= old_scale; o0.z *= old_scale; o0.w *= old_scale;
        o1.x *= old_scale; o1.y *= old_scale; o1.z *= old_scale; o1.w *= old_scale;
        o2.x *= old_scale; o2.y *= old_scale; o2.z *= old_scale; o2.w *= old_scale;
        o3.x *= old_scale; o3.y *= old_scale; o3.z *= old_scale; o3.w *= old_scale;

        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

/* Same cliff as the indexed kernel above, found by the occupancy audit: 70
 * registers x 256 threads = 17920 per block against 65536 per SM, so 3 blocks
 * fit and occupancy sits at 50%.  smem (8389 B) allows 12 blocks, so registers
 * alone are binding here -- capping at 64 fits a 4th block (66.7%).  This is
 * 299 ms of a ~4.6 s prefill.  Launched <<<grid, 256>>> at both call sites. */
#ifndef PULSAR_ATTN_STATIC_MIN_BLOCKS
#define PULSAR_ATTN_STATIC_MIN_BLOCKS 4
#endif

/* The release line's occupancy cap (__launch_bounds__, +8.9% prefill,
 * bit-exact -- 1d2ef4f).  The cap stays on the templated form: it is the
 * thing that bought the +8.9%, and dropping it while changing the Q load
 * would confound two effects. */
template <typename QT>
__global__ __launch_bounds__(256, PULSAR_ATTN_STATIC_MIN_BLOCKS)
static void attention_decode_mixed_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const QT *q,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t non_causal,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,
        const int32_t * __restrict__ positions,
        const int32_t * __restrict__ seq_id,
        const void * const * __restrict__ comp_bank_ptrs,
        uint32_t comp_cap,
        uint32_t n_banks,
        float * __restrict__ part_o,
        float * __restrict__ part_ml) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * 8u + warp;
    const bool valid_head = head < n_head;
    /* Split-KV mode (gridDim.z > 1): each z-block walks only its slice of the
     * row list and emits UNNORMALIZED partials (m, l, o) for a separate merge
     * kernel; the sink joins at merge time, once per head.  gridDim.z == 1
     * (part_o/part_ml NULL) is the classic single-walk kernel, bit-identical
     * to before this parameter existed: the slice below degenerates to
     * [0, n_score) and the epilogue takes the old path. */
    const uint32_t n_split = gridDim.z;
    const uint32_t split = blockIdx.z;
    if (seq_id && (uint32_t)seq_id[t] >= n_banks) {
        /* Dead/evicted row: see attention_decode_mixed_kernel.  All threads
         * return together (no __syncthreads has run yet).  In split mode the
         * merge kernel reads every partial slot, so a dead row must write
         * m=-inf/l=0/o=0 partials rather than leave stale scratch: the merge
         * then yields exactly 0, matching the direct zeroing below. */
        if (valid_head) {
            if (n_split > 1u) {
                const uint64_t pbase = ((uint64_t)t * n_head + head) * n_split + split;
                float4 *po = (float4 *)(part_o + pbase * head_dim);
                const float4 z4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                po[lane +  0u] = z4;
                po[lane + 32u] = z4;
                po[lane + 64u] = z4;
                po[lane + 96u] = z4;
                if (lane == 0) {
                    part_ml[pbase * 2u + 0u] = -INFINITY;
                    part_ml[pbase * 2u + 1u] = 0.0f;
                }
            } else {
                float *oh = heads + ((uint64_t)t * n_head + head) * head_dim;
                for (uint32_t d = lane; d < head_dim; d += 32u) oh[d] = 0.0f;
            }
        }
        return;
    }

    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t raw_count_s;
    __shared__ uint32_t raw_first_idx_s;
    __shared__ float4 kv_shared[4 * 128];

    /* Descriptor preamble: see attention_decode_mixed_kernel. */
    const uint32_t qpos = positions ? (uint32_t)positions[t] : pos0 + t;
    uint32_t eff_n_raw = n_raw;
    uint32_t eff_raw_start = raw_start;
    uint32_t first_raw_pos;
    if (positions) {
        eff_n_raw = (window != 0u && qpos + 1u > window) ? window : qpos + 1u;
        if (eff_n_raw > raw_cap) eff_n_raw = raw_cap;
        eff_raw_start = (qpos + 1u - eff_n_raw) % raw_cap;
        first_raw_pos = qpos + 1u - eff_n_raw;
    } else {
        first_raw_pos = pos0 + n_tokens - n_raw;
    }
    const uint32_t raw_base = seq_id ? (uint32_t)seq_id[t] * raw_cap : 0u;
    /* Per-bank comp base (see attention_decode_mixed_kernel). */
    const uint32_t sid_b = seq_id ? (uint32_t)seq_id[t] : 0u;
    const float *comp_src = comp_bank_ptrs ? (const float *)comp_bank_ptrs[sid_b] : comp_kv;
    const uint64_t comp_base = comp_bank_ptrs ? 0u
                             : (seq_id ? (uint64_t)sid_b * comp_cap : 0u);
    /* Scalar decode-entry convention: n_tokens==1, pos0==0, ratio==0,
     * positions==NULL — the caller means "attend to ALL cached rows" and does
     * NOT supply a real absolute position, so qpos is 0 and the windowing
     * arithmetic below does not apply.  Mirrors attention_decode_mixed_kernel. */
    const bool single_all = (n_tokens == 1u && ratio == 0u && positions == NULL);
    uint32_t comp_count = 0;
    if (n_comp != 0u) {
        if (single_all) {
            comp_count = n_comp;
        } else if (ratio != 0u) {
            comp_count = (qpos + 1u) / ratio;
            if (comp_count > n_comp) comp_count = n_comp;
        }
    }
    if (threadIdx.x == 0) {
        uint32_t raw_count = 0;
        uint32_t raw_first_idx = 0;
        if (eff_n_raw != 0u) {
            const uint32_t raw_last_pos = first_raw_pos + eff_n_raw - 1u;
            /* single_all MUST be handled before the qpos comparison below.  In
             * that convention first_raw_pos = pos0 + n_tokens - n_raw = 1 - 128,
             * which UNDERFLOWS to 4294967169, so `qpos(0) >= first_raw_pos` is
             * false and raw_count would stay 0 — silently dropping the ENTIRE
             * SWA raw window while comp_count above stayed correct.  That is not
             * a visible failure: it just answers from compressed rows only, and
             * measures as a large FAKE speedup (skipping 128 rows/layer/token)
             * with heavy logit damage underneath. */
            if (single_all) {
                raw_count = eff_n_raw > 256u ? 256u : eff_n_raw;
            } else if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0u && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = non_causal ? raw_last_pos : (qpos < raw_last_pos ? qpos : raw_last_pos);
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
        raw_count_s = raw_count;
        raw_first_idx_s = raw_first_idx;
    }
    __syncthreads();
    const uint32_t raw_count = raw_count_s;
    const uint32_t raw_first_idx = raw_first_idx_s;
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = raw_base + (eff_raw_start + raw_first_idx + r) % raw_cap;
    }
    __syncthreads();

    const uint32_t n_score = raw_count + comp_count;
    /* This block's row slice: ceil-divided so the last split absorbs the
     * remainder; an empty slice ([lo >= n_score)) skips the walk and emits
     * the m=-inf/l=0 identity partial.  n_split==1 => [0, n_score). */
    const uint32_t rows_per_split = (n_score + n_split - 1u) / n_split;
    const uint32_t row_lo = split * rows_per_split;
    uint32_t row_hi = row_lo + rows_per_split;
    if (row_hi > n_score) row_hi = n_score;
    const float scale = rsqrtf((float)head_dim);
    const QT *qrow = valid_head
        ? (q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q_load4<QT>(qrow, lane +  0u);
        q1 = q_load4<QT>(qrow, lane + 32u);
        q2 = q_load4<QT>(qrow, lane + 64u);
        q3 = q_load4<QT>(qrow, lane + 96u);
    }

    float max_s = -INFINITY;
    float sum_s = 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t row0 = row_lo; row0 < row_hi; row0 += 4u) {
        const uint32_t nr = row_hi - row0 < 4u ? row_hi - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            if (sr < raw_count) {
                kv_shared[off] = raw_kv_ld4(raw_kv, (uint64_t)raw_rows[sr], c4, head_dim);
            } else {
                const uint32_t n_nope = head_dim - PULSAR_ATTN_PACK_NROT;
                const uint8_t *pr = (const uint8_t *)comp_src +
                    (comp_base + (sr - raw_count)) * PULSAR_ATTN_PACK_ROWBYTES(head_dim);
                const uint32_t base = c4 << 2;
                float4 v;
                if (base < n_nope) {
                    /* base%4 == 0 and blocks are 64-aligned, so the four dims
                     * share one scale and one aligned uint32 of e4m3 bytes. */
                    const float scale = __uint_as_float((uint32_t)pr[n_nope + (base / PULSAR_FP8_KV_BLOCK)] << 23);
                    const uint32_t w = *(const uint32_t *)(pr + base);
                    v.x = attn_pack_e4m3(w & 0xffu, scale);
                    v.y = attn_pack_e4m3((w >> 8) & 0xffu, scale);
                    v.z = attn_pack_e4m3((w >> 16) & 0xffu, scale);
                    v.w = attn_pack_e4m3(w >> 24, scale);
                } else {
                    const __nv_bfloat16 *rope = (const __nv_bfloat16 *)(pr + n_nope + PULSAR_ATTN_PACK_SCALES_PAD(head_dim));
                    v.x = __bfloat162float(rope[base - n_nope + 0u]);
                    v.y = __bfloat162float(rope[base - n_nope + 1u]);
                    v.z = __bfloat162float(rope[base - n_nope + 2u]);
                    v.w = __bfloat162float(rope[base - n_nope + 3u]);
                }
                kv_shared[off] = v;
            }
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) +
                              dot4_f32(q1, k1) +
                              dot4_f32(q2, k2) +
                              dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(0xffffffffu, score, 0);

                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m);
                const float row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + k0.x * row_scale;
                o0.y = o0.y * old_scale + k0.y * row_scale;
                o0.z = o0.z * old_scale + k0.z * row_scale;
                o0.w = o0.w * old_scale + k0.w * row_scale;
                o1.x = o1.x * old_scale + k1.x * row_scale;
                o1.y = o1.y * old_scale + k1.y * row_scale;
                o1.z = o1.z * old_scale + k1.z * row_scale;
                o1.w = o1.w * old_scale + k1.w * row_scale;
                o2.x = o2.x * old_scale + k2.x * row_scale;
                o2.y = o2.y * old_scale + k2.y * row_scale;
                o2.z = o2.z * old_scale + k2.z * row_scale;
                o2.w = o2.w * old_scale + k2.w * row_scale;
                o3.x = o3.x * old_scale + k3.x * row_scale;
                o3.y = o3.y * old_scale + k3.y * row_scale;
                o3.z = o3.z * old_scale + k3.z * row_scale;
                o3.w = o3.w * old_scale + k3.w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }

    if (valid_head && n_split > 1u) {
        /* Split mode: emit the raw online-softmax state for the merge kernel.
         * o is the UNNORMALIZED sum exp(s - m)*v over this slice; the sink
         * joins at merge, once per head.  An empty slice emits the identity
         * (m=-inf, l=0, o=0). */
        const uint64_t pbase = ((uint64_t)t * n_head + head) * n_split + split;
        float4 *po = (float4 *)(part_o + pbase * head_dim);
        po[lane +  0u] = o0;
        po[lane + 32u] = o1;
        po[lane + 64u] = o2;
        po[lane + 96u] = o3;
        if (lane == 0) {
            part_ml[pbase * 2u + 0u] = max_s;
            part_ml[pbase * 2u + 1u] = sum_s;
        }
        return;
    }
    if (valid_head) {
        const float sink = sinks[head];
        const float new_m = fmaxf(max_s, sink);
        const float old_scale = expf(max_s - new_m);
        const float sink_scale = expf(sink - new_m);
        sum_s = sum_s * old_scale + sink_scale;
        o0.x *= old_scale; o0.y *= old_scale; o0.z *= old_scale; o0.w *= old_scale;
        o1.x *= old_scale; o1.y *= old_scale; o1.z *= old_scale; o1.w *= old_scale;
        o2.x *= old_scale; o2.y *= old_scale; o2.z *= old_scale; o2.w *= old_scale;
        o3.x *= old_scale; o3.y *= old_scale; o3.z *= old_scale; o3.w *= old_scale;

        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

/* Softmax merge for the split-KV decode walk: combines the per-slice
 * online-softmax partials (m_i, l_i, o_i) of one (token, head) with the
 * standard log-sum-exp reassociation and applies the sink term exactly once:
 *
 *   m*  = max(max_i m_i, sink)
 *   l*  = sum_i l_i*exp(m_i - m*) + exp(sink - m*)
 *   out = sum_i o_i*exp(m_i - m*) / l*
 *
 * Empty or dead slices carry (m=-inf, l=0, o=0) and vanish: exp(-inf - m*)
 * is 0 because m* >= sink is finite.  A dead row (every slice the identity)
 * therefore yields exactly 0, matching the direct kernel's zeroing.  NOT
 * bit-identical to the single-walk kernel -- the summation is reassociated;
 * same numerics class as the heads8-online carve-out itself, gated the same
 * way (engine KL closed loop, see docs/engine-perf-map.md).
 * Launch: grid (n_tokens, n_head), 128 threads; each thread owns 4 dims. */
__global__ static void attention_decode_split_merge_kernel(
        float *heads,
        const float *sinks,
        const float * __restrict__ part_o,
        const float * __restrict__ part_ml,
        uint32_t n_split,
        uint32_t n_head,
        uint32_t head_dim) {
    const uint32_t t = blockIdx.x;
    const uint32_t head = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (head >= n_head || head_dim != 512u) return;
    const uint64_t pbase = ((uint64_t)t * n_head + head) * n_split;

    const float sink = sinks[head];
    float m_star = sink;
    for (uint32_t i = 0; i < n_split; i++)
        m_star = fmaxf(m_star, part_ml[(pbase + i) * 2u + 0u]);
    float l_star = expf(sink - m_star);
    float w[8];
    for (uint32_t i = 0; i < n_split && i < 8u; i++) {
        w[i] = expf(part_ml[(pbase + i) * 2u + 0u] - m_star);
        l_star += part_ml[(pbase + i) * 2u + 1u] * w[i];
    }
    const float inv = l_star == 0.0f ? 0.0f : 1.0f / l_star;

    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (uint32_t i = 0; i < n_split && i < 8u; i++) {
        const float4 v = ((const float4 *)(part_o + (pbase + i) * head_dim))[tid];
        acc.x += w[i] * v.x;
        acc.y += w[i] * v.y;
        acc.z += w[i] * v.z;
        acc.w += w[i] * v.w;
    }
    acc.x *= inv; acc.y *= inv; acc.z *= inv; acc.w *= inv;
    ((float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim))[tid] = acc;
}

/* Split-KV partial scratch: static device storage so the split launch is
 * CUDA-graph-safe (no allocation at capture or replay time) and needs no
 * per-call setup.  Sized for the worst gated shape (8 tokens x 128 heads x
 * 8 splits x 512 dims = 16.8 MiB).  The decode step executes one attention
 * launch at a time on one stream, so a single scratch is safe; a future
 * multi-stream decode would need per-stream scratch. */
#define PULSAR_DEC_SPLITKV_S 8u
#define PULSAR_DEC_SPLITKV_MAX_TOKENS 8u
#define PULSAR_DEC_SPLITKV_MAX_HEADS 128u
/* CONCURRENCY (multi-stream decode, mid-roadmap): a single shared __device__
 * partials buffer is safe only while one thread submits decode. It CANNOT be
 * made thread_local (device globals are per-context, not per-host-thread); a
 * second concurrent attention launch would clobber these partials. The fix is
 * per-stream/per-call scratch, not a keyword change. */
static __device__ float g_dec_splitkv_part_o[PULSAR_DEC_SPLITKV_MAX_TOKENS *
        PULSAR_DEC_SPLITKV_MAX_HEADS * PULSAR_DEC_SPLITKV_S * 512u];
static __device__ float g_dec_splitkv_part_ml[PULSAR_DEC_SPLITKV_MAX_TOKENS *
        PULSAR_DEC_SPLITKV_MAX_HEADS * PULSAR_DEC_SPLITKV_S * 2u];

static int attention_decode_heads8_launch(
        float *heads, const float *sinks, const pulsar_q_t *q, const float *raw_kv,
        const float *comp_kv, uint32_t non_causal, uint32_t n_tokens, uint32_t pos0, uint32_t n_raw,
        uint32_t raw_cap, uint32_t raw_start, uint32_t n_comp, uint32_t window,
        uint32_t ratio, uint32_t n_head, uint32_t head_dim,
        const int32_t *positions, const int32_t *seq_id,
        const void * const *comp_bank_ptrs, uint32_t comp_cap, uint32_t n_banks,
        const char *what) {
    static const int use_split = pulsar_env_tier_on("PULSAR_CUDA_DECODE_SPLITKV");
    if (use_split && n_tokens <= PULSAR_DEC_SPLITKV_MAX_TOKENS &&
        n_head <= PULSAR_DEC_SPLITKV_MAX_HEADS && head_dim == 512u) {
        static int announced = 0;
        if (!announced) {
            announced = 1;
            fprintf(stderr, "pulsar: decode attention = split-KV x%u tier "
                            "(default; PULSAR_CUDA_DECODE_SPLITKV=0 opts out; "
                            "softmax-merge reassociation)\n",
                    PULSAR_DEC_SPLITKV_S);
        }
        static float *part_o = NULL;
        static float *part_ml = NULL;
        if (!part_o &&
            (cudaGetSymbolAddress((void **)&part_o, g_dec_splitkv_part_o) != cudaSuccess ||
             cudaGetSymbolAddress((void **)&part_ml, g_dec_splitkv_part_ml) != cudaSuccess)) {
            part_o = NULL;
            fprintf(stderr, "pulsar: split-KV scratch symbol lookup FAILED; "
                            "refusing to fall through\n");
            return 0;
        }
        dim3 sgrid(n_tokens, (n_head + 7u) / 8u, PULSAR_DEC_SPLITKV_S);
        attention_decode_mixed_heads8_online_kernel<<<sgrid, 256>>>(heads, sinks,
                q, raw_kv, comp_kv, non_causal,
                n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp, window, ratio,
                n_head, head_dim, positions, seq_id, comp_bank_ptrs,
                comp_cap, n_banks, part_o, part_ml);
        dim3 mgrid(n_tokens, n_head, 1);
        attention_decode_split_merge_kernel<<<mgrid, 128>>>(heads, sinks,
                part_o, part_ml, PULSAR_DEC_SPLITKV_S, n_head, head_dim);
        return cuda_ok(cudaGetLastError(), what);
    }
    dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
    attention_decode_mixed_heads8_online_kernel<<<grid, 256>>>(heads, sinks,
            q, raw_kv, comp_kv, non_causal,
            n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp, window, ratio,
            n_head, head_dim, positions, seq_id, comp_bank_ptrs,
            comp_cap, n_banks, NULL, NULL);
    return cuda_ok(cudaGetLastError(), what);
}

__global__ static void indexed_topk_sort_512_asc_kernel(
        int32_t *dst,
        const int32_t *src,
        uint32_t n_tokens) {
    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= 512u) return;
    __shared__ int32_t rows[512];

    const int32_t *src_row = src + (uint64_t)t * 512u;
    int32_t *dst_row = dst + (uint64_t)t * 512u;
    rows[tid] = src_row[tid];
    __syncthreads();

    for (uint32_t k = 2u; k <= 512u; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            const uint32_t other = tid ^ j;
            if (other > tid && other < 512u) {
                const int32_t a = rows[tid];
                const int32_t b = rows[other];
                const bool up = (tid & k) == 0u;
                if ((up && a > b) || (!up && a < b)) {
                    rows[tid] = b;
                    rows[other] = a;
                }
            }
            __syncthreads();
        }
    }

    dst_row[tid] = rows[tid];
}

int pulsar_gpu_attention_decode_heads_tensor(
        pulsar_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *raw_kv,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        const pulsar_gpu_tensor *comp_kv,
        uint32_t                n_comp,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (        !heads || !q || !raw_kv || !model_map || n_raw == 0 || raw_cap < n_raw ||
        raw_start >= raw_cap || (n_comp != 0 && !comp_kv) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_head * head_dim * PULSAR_Q_ELT_SIZE ||
        raw_kv->bytes < (uint64_t)raw_cap * PULSAR_ATTN_PACK_ROWBYTES(head_dim) ||
        head_dim <= PULSAR_ATTN_PACK_NROT ||
        ((head_dim - PULSAR_ATTN_PACK_NROT) % PULSAR_FP8_KV_BLOCK) != 0 ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp *
         (PULSAR_ATTN_PACK_ROWBYTES(head_dim))) ||
        false) {
        return 0;
    }
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    /* Single-token decode routes to the heads8-online kernel by DEFAULT
     * (2026-08-02), not only on score-buffer overflow.  The generic
     * per-(row,head) kernel re-walks every raw+comp row once PER HEAD —
     * 64 independent row scans per layer per token — and profiled at
     * 10.9 ms/token @ctx2048 (~27x its byte roofline; 18% of decode).  The
     * heads8-online kernel stages each row once for 8 heads, the same
     * restructuring the indexed decode carve-out shipped earlier ("~5x over
     * the generic per-(row,head) kernel").  NOT bit-exact vs the generic
     * kernel (online-softmax fold order; reassociation class, same as that
     * carve-out) — this is a perplexity/eval-gated numerics change. */
    if (!cuda_attention_score_buffer_fits(n_comp) || head_dim == 512u) {
        if (head_dim == 512u) {
            return attention_decode_heads8_launch((float *)heads->ptr, sinks,
                    (const pulsar_q_t *)q->ptr, (const float *)raw_kv->ptr,
                    n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                    0, 1, 0, n_raw, raw_cap,
                    raw_start, n_comp, 0, 0, n_head, head_dim,
                    NULL, NULL, NULL, 0, 1,
                    "attention decode online launch");
        }
        fprintf(stderr, "pulsar: CUDA attention score buffer too small for %u compressed rows\n", n_comp);
        return 0;
    }
    dim3 grid(1, n_head, 1);
    attention_decode_mixed_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                 sinks,
                                                 (const pulsar_q_t *)q->ptr,
                                                 (const float *)raw_kv->ptr,
                                                 n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                 0,
                                                 1, 0, n_raw, raw_cap, raw_start, n_comp,
                                                 0, 0, n_head, head_dim,
                                                 NULL, NULL, NULL, 0, 1);
    return cuda_ok(cudaGetLastError(), "attention decode launch");
}

/* L037 lever 3 fallback: a wrapper received RAW q (q_prep != NULL) but is
 * about to hand it to a NON-f16 consumer -- apply the standalone norm+rope
 * now and continue exactly as the pre-fusion code did. Placed immediately
 * above every such consumer; the fused f16 calls pass q_prep through
 * instead. A missed site is a wrong answer the prefill/frontier gates catch,
 * not a slow path. */
static int attention_q_prep_apply(const pulsar_gpu_tensor *q, uint32_t n_tokens,
                                  uint32_t n_head, uint32_t head_dim,
                                  uint32_t pos0, const pulsar_gpu_tensor *positions,
                                  const pulsar_gpu_q_prep *qp) {
    return pulsar_gpu_head_rms_norm_rope_tail_tensor((pulsar_gpu_tensor *)q,
            n_tokens, n_head, head_dim, qp->n_rot, pos0, qp->n_ctx_orig, false,
            qp->freq_base, qp->freq_scale, qp->ext_factor, qp->attn_factor,
            qp->beta_fast, qp->beta_slow, qp->eps, positions,
                                                   /* q_f16: */ 0);
}
#define ATTN_Q_PREP_FALLBACK(qt, ntok, pos0v, posv) \
    do { if (q_prep) { \
        if (!attention_q_prep_apply((qt), (ntok), n_head, head_dim, (pos0v), (posv), q_prep)) \
            return 0; \
        q_prep = NULL; \
    } } while (0)

int pulsar_gpu_attention_prefill_raw_heads_mx_tensor(pulsar_gpu_tensor *heads, const void *model_map, uint64_t model_size, uint64_t sinks_offset, const pulsar_gpu_tensor *q, const pulsar_gpu_tensor *raw_kv, uint32_t n_tokens, uint32_t window, uint32_t n_head, uint32_t head_dim,
        void *gact_data, void *gact_scale, int gact_kbp, uint32_t gact_slab, uint32_t n_groups, uint32_t n_nope, int *mx_out,
        const pulsar_gpu_tensor *positions, const pulsar_gpu_q_prep *q_prep) {
    if (mx_out) *mx_out = 0;
    if (!heads || !q || !raw_kv || !model_map || sinks_offset > model_size ||
        model_size - sinks_offset < (uint64_t)n_head * sizeof(float) ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * PULSAR_Q_ELT_SIZE ||
        raw_kv->bytes < (uint64_t)n_tokens * PULSAR_ATTN_PACK_ROWBYTES(head_dim) ||
        window > 256) return 0;
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    /* The dense f16 launch does not carry per-token positions, so a banked
     * batch that somehow routes here must norm the old way (rope by t would
     * be silently wrong). Dense zero-prefix batches pass positions == NULL. */
    if (positions) ATTN_Q_PREP_FALLBACK(q, n_tokens, 0u, positions);
    /* One-shot branch report, same as the mixed path. This is the RAW entry;
     * pulsar-bench showed real prefill never reaches the mixed launch, so this
     * is where production prefill attention is actually served. */
    static int raw_path_reported = 0;
    if (!raw_path_reported) {
        raw_path_reported = 1;
        const int takes_window = n_tokens > 1 && head_dim == 512;
        fprintf(stderr,
                "pulsar: ATTN-RAW n_tokens=%u head_dim=%u window=%u -> %s\n",
                n_tokens, head_dim, window,
                takes_window ? "FUSED window kernel (no score matrix)"
                             : (g_cublas_ready
                                    ? "unfused cuBLAS two-GEMM"
                                    : "generic per-token kernel"));
    }
    if (n_tokens > 1 && head_dim == 512) {
        /* fp16 tensor-core tier.  The kernel this replaces runs at pipe_tensor
         * 0%; see docs/engine-perf-map.md.  DEFAULT-ON since 2026-08-08
         * (PULSAR_CUDA_ATTN_F16=0 opts out): fp16 operands change the numbers,
         * and the suite-v1 KL run is what cleared the flip — the component
         * measurement (tests/attn_precision_fidelity.cc: top-1 attention
         * position preserved ~100%, KL <= 3e-7) was evidence, not the gate.
         * Shape conditions are checked by the launcher itself, and a 0 return
         * here is a REAL failure, so it is reported rather than silently
         * demoted to the FMA kernel. */
        static const int use_f16_attn = pulsar_env_tier_on("PULSAR_CUDA_ATTN_F16");
        if (use_f16_attn && head_dim == 512u && (n_head % 16u) == 0u) {
            static int announced = 0;
            if (!announced) {
                announced = 1;
                fprintf(stderr, "pulsar: attention = fp16 tensor-core tier "
                                "(default; PULSAR_CUDA_ATTN_F16=0 opts out; operands rounded to fp16)\n");
            }
            /* Only the fp16 tier can emit the grouped E4M3 encoding.  If it
             * declines the batch we fall through WITHOUT setting *mx_out, so
             * the caller does not note() and the "a" GEMM runs its own
             * quantiser -- a half-written encoding would be a wrong answer,
             * not a slow one. */
            if (pulsar_gpu_attention_f16_prefill_mx(
                    (float *)heads->ptr, sinks, (const pulsar_q_t *)q->ptr,
                    (const float *)raw_kv->ptr, NULL,
                    n_tokens, 0u, window, 1u, n_head, head_dim,
                    gact_data, gact_scale, gact_kbp, gact_slab, n_groups, n_nope,
                    0u, n_tokens, q_prep)) {
                if (mx_out && gact_data) *mx_out = 1;
                return 1;
            }
            fprintf(stderr, "pulsar: fp16 attention tier FAILED; refusing to "
                            "fall through to the f32 kernel\n");
            return 0;
        }
    }
    ATTN_Q_PREP_FALLBACK(q, n_tokens, 0u, NULL);
    if (g_cublas_ready && n_tokens > 1 && head_dim == 512) {
        const uint32_t n_keys = n_tokens;
        const uint64_t score_count = (uint64_t)n_head * n_tokens * n_keys;
        const uint64_t out_count = (uint64_t)n_head * n_tokens * head_dim;
        const uint64_t score_bytes = score_count * sizeof(float);
        /* cuBLAS consumes f32 operands: in f16 raw mode, expand the raw rows
         * into an f32 staging region first (pure conversion, exact values, so
         * the gemms see the identical matrix the f32 path reads). */
        const uint64_t kv_count = (uint64_t)n_tokens * head_dim;
        const uint64_t kv_bytes = kv_count * sizeof(float);
        const uint64_t out_bytes = out_count * sizeof(float);
        /* The score GEMM's operands must agree in type, so a narrowed Q needs an
         * f16 KV operand.  Staging KV (n_keys*head_dim, stride 0 across heads)
         * rather than widening Q (n_tok*n_head*head_dim) is ~128x less work:
         * 4 MiB against 512 MiB at a 4096-token prefill.  It is also EXACT --
         * the packed KV is E4M3 with a power-of-two block scale, so every value
         * fits f16's mantissa and range with nothing to round.
         * The value GEMM below keeps the f32 copy: its B operand is post-softmax
         * P, and narrowing THAT would be a real precision change. */
        const uint64_t kv16_bytes = (sizeof(pulsar_q_t) == sizeof(float))
                                  ? 0ull : kv_count * sizeof(__half);
        const uint64_t tmp_bytes = ((kv_bytes + 255u) & ~255ull) +
                                   ((score_bytes + 255u) & ~255ull) + out_bytes +
                                   ((kv16_bytes + 255u) & ~255ull);
        cuda_arena ar;
        if (!cuda_arena_begin(&ar, tmp_bytes, "attention raw cublas")) return 0;
        float *tmp     = (float *)cuda_arena_take(&ar, kv_bytes, 256);
        float *scores  = (float *)cuda_arena_take(&ar, score_bytes, 256);
        float *out_tmp = (float *)cuda_arena_take(&ar, out_bytes, 256);
        if (!out_tmp) return 0;   /* take() latches: one check covers all three */
        __half *kv16 = NULL;
        if (kv16_bytes) {
            kv16 = (__half *)cuda_arena_take(&ar, kv16_bytes, 256);
            if (!kv16) return 0;
        }
        const float *kv_mat = (const float *)raw_kv->ptr;
        {
            attention_prefill_pack_mixed_kv_kernel<<<(kv_count + 255) / 256, 256>>>(
                    tmp,
                    (const float *)raw_kv->ptr,
                    (const float *)raw_kv->ptr,
                    n_tokens,
                    0,
                    head_dim,
                    NULL);
            if (!cuda_ok(cudaGetLastError(), "attention raw f16 expand launch")) return 0;
            kv_mat = tmp;
        }
        const void *score_a = kv_mat;
        if (kv16) {
            attention_prefill_pack_mixed_kv_kernel<<<(kv_count + 255) / 256, 256>>>(
                    NULL,
                    (const float *)raw_kv->ptr,
                    (const float *)raw_kv->ptr,
                    n_tokens,
                    0,
                    head_dim,
                    kv16);
            if (!cuda_ok(cudaGetLastError(), "attention raw kv f16 pack launch")) return 0;
            score_a = kv16;
        }
        const float alpha = rsqrtf((float)head_dim);
        const float beta = 0.0f;
        /* Ex rather than Sgemm so the operands follow the stored Q type.  The
         * accumulator and output stay f32; only the operand encoding moves. */
        cublasStatus_t st = cublasGemmStridedBatchedEx(g_cublas,
                CUBLAS_OP_T, CUBLAS_OP_N,
                (int)n_keys, (int)n_tokens, (int)head_dim,
                &alpha,
                score_a,      PULSAR_Q_CUDA_TYPE, (int)head_dim, (long long)0,
                q->ptr,       PULSAR_Q_CUDA_TYPE, (int)(n_head * head_dim), (long long)head_dim,
                &beta,
                scores,       CUDA_R_32F,         (int)n_keys, (long long)n_keys * n_tokens,
                (int)n_head,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (!cublas_ok(st, "attention raw score gemm")) return 0;
        dim3 sgrid(n_tokens, n_head, 1);
        attention_prefill_raw_softmax_kernel<<<sgrid, 256>>>(scores, sinks, n_tokens, window, n_keys);
        if (!cuda_ok(cudaGetLastError(), "attention raw softmax launch")) return 0;
        const float one = 1.0f;
        st = cublasSgemmStridedBatched(g_cublas,
                                       CUBLAS_OP_N,
                                       CUBLAS_OP_N,
                                       (int)head_dim,
                                       (int)n_tokens,
                                       (int)n_keys,
                                       &one,
                                       kv_mat,
                                       (int)head_dim,
                                       0,
                                       scores,
                                       (int)n_keys,
                                       (long long)n_keys * n_tokens,
                                       &beta,
                                       out_tmp,
                                       (int)head_dim,
                                       (long long)head_dim * n_tokens,
                                       (int)n_head);
        if (!cublas_ok(st, "attention raw value gemm")) return 0;
        uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
        attention_prefill_unpack_heads_kernel<<<(n + 255) / 256, 256>>>((float *)heads->ptr,
                                                                        out_tmp,
                                                                        n_tokens,
                                                                        n_head,
                                                                        head_dim);
        return cuda_ok(cudaGetLastError(), "attention raw unpack launch");
    }
    dim3 grid(n_tokens, n_head, 1);
    attention_prefill_raw_kernel<<<grid, 128>>>((float *)heads->ptr,
                                                sinks,
                                                (const pulsar_q_t *)q->ptr,
                                                (const float *)raw_kv->ptr,
                                                n_tokens, window, n_head, head_dim);
    return cuda_ok(cudaGetLastError(), "attention_prefill_raw launch");
}

static int attention_decode_batch_launch(
        pulsar_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *raw_kv,
        const pulsar_gpu_tensor *comp_kv,
        uint32_t                non_causal,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim,
        const pulsar_gpu_tensor *positions,
        const pulsar_gpu_tensor *seq_id,
        const pulsar_gpu_tensor *comp_bank_ptrs,
        uint32_t                comp_cap,
        uint32_t                n_banks,
        const pulsar_gpu_q_prep *q_prep) {
    /* Descriptor (banked) mode: both per-row arrays or neither; the KV
     * operands are whole bank pools, so byte bounds scale by n_banks and the
     * uint32 row ABI (seq*cap + local) must not overflow.  The scalar
     * n_raw/raw_start are IGNORED and NOT validated in this mode (pass 0):
     * the raw span and ring start are per-row derived in the kernels from
     * positions[t].  raw_cap must still be the true per-bank ring capacity
     * (it addresses the ring and sizes the byte bound); scalar n_comp is the
     * cross-bank superset clamp (comp_cap is the per-bank row stride, so it
     * must cover it).  All rejections here are fail-loud: a silent 0 return
     * looks like a generic launch failure to the driver. */
    const int descr = positions != NULL || seq_id != NULL;
    if (descr &&
        (!positions || !seq_id || n_banks == 0 || raw_cap == 0 ||
         positions->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
         seq_id->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
         (n_comp != 0 && comp_cap < n_comp) ||
         /* Per-row derivation clamps the raw span to `window`, and the
          * kernels' raw_rows scratch holds PULSAR_CUDA_ATTENTION_RAW_SCORE_CAP
          * rows.  A zero (unbounded) or over-cap window would be silently
          * truncated to the OLDEST rows — fail loud instead. */
         window == 0u || window > PULSAR_CUDA_ATTENTION_RAW_SCORE_CAP ||
         (uint64_t)n_banks * raw_cap > 4294967296ull ||
         (uint64_t)n_banks * comp_cap > 4294967296ull)) {
        fprintf(stderr,
                "pulsar: banked decode attention rejected: bad descriptor args "
                "(n_tokens=%u n_banks=%u raw_cap=%u comp_cap=%u n_comp=%u window=%u)\n",
                n_tokens, n_banks, raw_cap, comp_cap, n_comp, window);
        return 0;
    }
    const uint64_t kv_banks = descr ? n_banks : 1u;
    const uint32_t kernel_n_banks = descr ? n_banks : 1u;
    /* comp_kv is the per-bank comp cache. With split allocations + a base-pointer
     * table (increment 2a) each bank is its OWN comp_cap-row allocation, and
     * comp_kv is bank 0's — so the byte bound is per-bank (comp_cap), not
     * n_banks*comp_cap. Without a table (legacy contiguous), it spans all banks. */
    const uint64_t comp_rows_min = (descr && comp_bank_ptrs) ? (uint64_t)comp_cap
                                 : descr ? (uint64_t)n_banks * comp_cap
                                         : (uint64_t)n_comp;
    if (        !heads || !q || !raw_kv || !model_map || n_tokens == 0 ||
        (!descr && (n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap)) ||
        (n_comp != 0 && !comp_kv) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * PULSAR_Q_ELT_SIZE ||
        raw_kv->bytes < kv_banks * raw_cap * PULSAR_ATTN_PACK_ROWBYTES(head_dim) ||
        head_dim <= PULSAR_ATTN_PACK_NROT ||
        ((head_dim - PULSAR_ATTN_PACK_NROT) % PULSAR_FP8_KV_BLOCK) != 0 ||
        (n_comp && comp_kv->bytes < comp_rows_min *
         (PULSAR_ATTN_PACK_ROWBYTES(head_dim))) ||
        false) {
        return 0;
    }
    if (n_comp != 0 && ratio == 0) return 0;
    const int32_t *positions_ptr = descr ? (const int32_t *)positions->ptr : NULL;
    const int32_t *seq_id_ptr = descr ? (const int32_t *)seq_id->ptr : NULL;
    /* Per-bank comp base-pointer table (descriptor mode only; NULL → the kernel's
     * scalar base + seq_id*comp_cap fallback, bit-identical to the contiguous pool). */
    const void * const *comp_bank_ptrs_ptr =
        (descr && comp_bank_ptrs) ? (const void * const *)comp_bank_ptrs->ptr : NULL;
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    if (!cuda_attention_score_buffer_fits(n_comp)) {
        ATTN_Q_PREP_FALLBACK(q, n_tokens, pos0, positions);
        if (head_dim == 512u) {
            return attention_decode_heads8_launch((float *)heads->ptr, sinks,
                    (const pulsar_q_t *)q->ptr, (const float *)raw_kv->ptr,
                    n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                    non_causal, n_tokens, pos0,
                    n_raw, raw_cap, raw_start, n_comp, window, ratio, n_head,
                    head_dim, positions_ptr, seq_id_ptr,
                    comp_bank_ptrs_ptr, comp_cap, kernel_n_banks,
                    "attention decode online launch");
        }
        fprintf(stderr, "pulsar: CUDA attention score buffer too small for %u compressed rows\n", n_comp);
        return 0;
    }
    if (n_tokens > 1 && head_dim == 512) {
        /* fp16 tensor-core tier for the CONTINUED-PREFILL batch: this is the
         * kernel that grows with context (27.9 ms/launch and 10.8% of GPU at a
         * 32k prefill) and it is token-parallel here (n_tokens >= 128), so the
         * fp16 decomposition applies.  True decode (n_tokens == 1) never
         * reaches this branch and stays on the f32 kernel.  Refused rather
         * than approximated: comp-mask, non-causal, and FP8 comp rows.  A 0
         * from the launcher is a real failure and is reported, not demoted. */
        if (pulsar_gpu_attention_prefill_reads_packed_comp() &&
            !non_causal && (n_head % 32u) == 0u) {
            static int announced_dc = 0;
            if (!announced_dc) {
                announced_dc = 1;
                fprintf(stderr, "pulsar: continued-prefill attention = fp16 tensor-core tier\n");
            }
            if (pulsar_gpu_attention_f16_indexed(
                    (float *)heads->ptr, sinks, (const pulsar_q_t *)q->ptr,
                    (const float *)raw_kv->ptr,
                    n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                    NULL /* no topk: visible-prefix sweep */,
                    n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp,
                    0u, window, ratio, n_head, head_dim,
                    (const int *)positions_ptr, (const int *)seq_id_ptr,
                    comp_bank_ptrs_ptr, comp_cap, kernel_n_banks, q_prep))
                return 1;
            fprintf(stderr, "pulsar: fp16 continued-prefill attention FAILED; "
                            "refusing to fall through\n");
            return 0;
        }
        ATTN_Q_PREP_FALLBACK(q, n_tokens, pos0, positions);
        return attention_decode_heads8_launch((float *)heads->ptr, sinks,
                (const pulsar_q_t *)q->ptr, (const float *)raw_kv->ptr,
                n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                non_causal, n_tokens, pos0,
                n_raw, raw_cap, raw_start, n_comp, window, ratio, n_head,
                head_dim, positions_ptr, seq_id_ptr,
                comp_bank_ptrs_ptr, comp_cap, kernel_n_banks,
                "attention decode window launch");
    }
    ATTN_Q_PREP_FALLBACK(q, n_tokens, pos0, positions);
    dim3 grid(n_tokens, n_head, 1);
    attention_decode_mixed_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                 sinks,
                                                 (const pulsar_q_t *)q->ptr,
                                                 (const float *)raw_kv->ptr,
                                                 n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                 non_causal, n_tokens, pos0, n_raw, raw_cap,
                                                 raw_start, n_comp, window, ratio, n_head, head_dim,
                                                 positions_ptr, seq_id_ptr, comp_bank_ptrs_ptr, comp_cap, kernel_n_banks);
    return cuda_ok(cudaGetLastError(), "attention decode batch launch");
}

int pulsar_gpu_attention_prefill_raw_heads_tensor(pulsar_gpu_tensor *heads, const void *model_map, uint64_t model_size, uint64_t sinks_offset, const pulsar_gpu_tensor *q, const pulsar_gpu_tensor *raw_kv, uint32_t n_tokens, uint32_t window, uint32_t n_head, uint32_t head_dim,
        const pulsar_gpu_tensor *positions, const pulsar_gpu_q_prep *q_prep) {
    return pulsar_gpu_attention_prefill_raw_heads_mx_tensor(heads, model_map, model_size, sinks_offset,
                                                            q, raw_kv, n_tokens, window, n_head, head_dim, NULL, NULL, 0, 0u, 0u, 0u, NULL,
                                                            positions, q_prep);
}

int pulsar_gpu_attention_decode_raw_batch_heads_tensor(
        pulsar_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                non_causal,
        const pulsar_gpu_tensor *positions,
        const pulsar_gpu_tensor *seq_id,
        uint32_t                comp_cap,
        uint32_t                n_banks,
        const pulsar_gpu_q_prep *q_prep) {
    return attention_decode_batch_launch(heads, model_map, model_size, sinks_offset,
                                      q, raw_kv, NULL, non_causal, n_tokens, pos0,
                                      n_raw, raw_cap, raw_start, 0, window, 1,
                                      n_head, head_dim,
                                      positions, seq_id, NULL /* raw path: no comp */, comp_cap, n_banks,
                                      q_prep);
}

int pulsar_gpu_attention_decode_mixed_batch_heads_tensor(
        pulsar_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *raw_kv,
        const pulsar_gpu_tensor *comp_kv,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                non_causal,
        const pulsar_gpu_tensor *positions,
        const pulsar_gpu_tensor *seq_id,
        const pulsar_gpu_tensor *comp_bank_ptrs,
        uint32_t                comp_cap,
        uint32_t                n_banks,
        const pulsar_gpu_q_prep *q_prep) {
    return attention_decode_batch_launch(heads, model_map, model_size, sinks_offset,
                                      q, raw_kv, comp_kv, non_causal,
                                      n_tokens, pos0, n_raw, raw_cap, raw_start,
                                      n_comp, window, ratio, n_head, head_dim,
                                      positions, seq_id, comp_bank_ptrs, comp_cap, n_banks,
                                      q_prep);
}

int pulsar_gpu_attention_indexed_mixed_batch_heads_tensor(
        pulsar_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *raw_kv,
        const pulsar_gpu_tensor *comp_kv,
        const pulsar_gpu_tensor *topk,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                top_k,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim,
        const pulsar_gpu_tensor *positions,
        const pulsar_gpu_tensor *seq_id,
        const pulsar_gpu_tensor *comp_bank_ptrs,
        uint32_t                comp_cap,
        uint32_t                n_banks,
        const pulsar_gpu_q_prep *q_prep) {
    /* Descriptor (banked) mode: same contract as attention_decode_batch_launch
     * (scalar n_raw/raw_start ignored and unvalidated, raw_cap must be the true
     * per-bank ring capacity, rejections fail-loud).  NOTE: banked rows are NO
     * LONGER forced onto the generic per-(row,head) kernel — 0e643eb taught the
     * heads8-online fast variant the banked descriptors, and the dispatch below
     * routes banked rows to it (see the rationale at the dispatch site).  That
     * recovered ~5x on ratio-4 banked prefill; do not "restore" the old
     * single-bank-only restriction. */
    const int descr = positions != NULL || seq_id != NULL;
    if (descr &&
        (!positions || !seq_id || n_banks == 0 || raw_cap == 0 || ratio == 0 ||
         positions->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
         seq_id->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
         comp_cap < n_comp ||
         /* raw_rows scratch bound — see attention_decode_batch_launch. */
         window == 0u || window > PULSAR_CUDA_ATTENTION_RAW_SCORE_CAP ||
         (uint64_t)n_banks * raw_cap > 4294967296ull ||
         (uint64_t)n_banks * comp_cap > 4294967296ull)) {
        fprintf(stderr,
                "pulsar: banked indexed attention rejected: bad descriptor args "
                "(n_tokens=%u n_banks=%u raw_cap=%u comp_cap=%u n_comp=%u window=%u ratio=%u)\n",
                n_tokens, n_banks, raw_cap, comp_cap, n_comp, window, ratio);
        return 0;
    }
    const uint64_t kv_banks = descr ? n_banks : 1u;
    /* Per-bank split: comp_kv is bank 0's comp_cap-row allocation when the base-
     * pointer table is present (see attention_decode_batch_launch). */
    const uint64_t comp_rows_min = (descr && comp_bank_ptrs) ? (uint64_t)comp_cap
                                 : descr ? (uint64_t)n_banks * comp_cap
                                         : (uint64_t)n_comp;
    if (        !heads || !q || !raw_kv || !comp_kv || !topk || !model_map ||
        n_tokens == 0 ||
        (!descr && (n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap)) ||
        n_comp == 0 || top_k == 0 ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * PULSAR_Q_ELT_SIZE ||
        raw_kv->bytes < kv_banks * raw_cap * PULSAR_ATTN_PACK_ROWBYTES(head_dim) ||
        head_dim <= PULSAR_ATTN_PACK_NROT ||
        ((head_dim - PULSAR_ATTN_PACK_NROT) % PULSAR_FP8_KV_BLOCK) != 0 ||
        comp_kv->bytes < comp_rows_min * (PULSAR_ATTN_PACK_ROWBYTES(head_dim)) ||
        topk->bytes < (uint64_t)n_tokens * top_k * sizeof(int32_t)) {
        return 0;
    }
    if (top_k > 512u) return 0;
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    const int32_t *positions_ptr = descr ? (const int32_t *)positions->ptr : NULL;
    const int32_t *seq_id_ptr = descr ? (const int32_t *)seq_id->ptr : NULL;
    const void * const *comp_bank_ptrs_ptr =
        (descr && comp_bank_ptrs) ? (const void * const *)comp_bank_ptrs->ptr : NULL;
    const int32_t *topk_ptr = (const int32_t *)topk->ptr;
    /* Launch-path dispatch flags: read the environment once per process. */
    static const int no_indexed_heads8 = getenv("PULSAR_CUDA_NO_INDEXED_HEADS8") != NULL;
    /* The sort stays OFF for n_tokens == 1.  It is a locality optimization:
     * attention_indexed_mixed_heads8_online_kernel reads topk[] in whatever order
     * it is given, clamps each id against visible_comp, and folds rows through an
     * online softmax that is correct for ANY permutation — there is no dedup, no
     * monotonicity assumption, and no binary search anywhere in it.  At decode the
     * whole 512-row compressed scan is L2-resident, so ascending row addresses buy
     * nothing while the bitonic sort costs ~4.2 us per token.
     *
     * It is NOT output-neutral, though: the online fold is order-dependent in
     * floating point, so sorting changes the reduction order and the logits with
     * it (measured 2026-08-13: max |Δlogit| 5.44 vs unsorted, for ~0% throughput
     * at frontiers 4k..32k).  "Locality only" describes the memory access, not
     * the arithmetic. */
    if (n_tokens > 1u && top_k == 512u) {
        const uint64_t sort_bytes = (uint64_t)n_tokens * top_k * sizeof(int32_t);
        int32_t *sorted = (int32_t *)cuda_tmp_alloc(sort_bytes, "indexed attention topk sort");
        if (!sorted) return 0;
        indexed_topk_sort_512_asc_kernel<<<n_tokens, 512>>>(sorted, topk_ptr, n_tokens);
        if (!cuda_ok(cudaGetLastError(), "indexed attention topk sort launch")) return 0;
        topk_ptr = sorted;
    }
    /* The heads8-online kernel is banked-aware (it takes the descriptor
     * arrays), so banked rows route here too — recovering the ~5x it holds over
     * the generic per-(row,head) kernel that banked mode was previously forced
     * onto.  The rb4 twopass variant stays single-bank-only, so a banked launch
     * always takes the online branch (never rb4).
     *
     * Scope guard: packed comp routes to the online kernel ONLY in banked mode
     * (descr) or on a single-token (decode) row.  Multi-token classic (non-descr)
     * paths keep their exact prior kernel choice — f32 -> online, pack -> generic.
     *
     * THE n_tokens == 1 CARVE-OUT CHANGES BEHAVIOUR, DELIBERATELY.  Single-token
     * decode used to fall through to attention_indexed_mixed_kernel at
     * grid(1, n_head): 64 blocks, each dequantizing the SAME compressed rows for
     * one head.  Routing it here instead gives grid(1, n_head/16): each block
     * stages a row into shared memory once and reuses it for 16 heads, amortizing
     * the pack dequant instructions and the load latency across the group.  This
     * is a LATENCY / instruction-amortization win, NOT a bandwidth win — the
     * indexer caps the compressed scan at top_k=512 rows (~0.46 MB per layer),
     * which sits entirely in GB10's 24 MiB L2, so the reads the old kernel
     * repeated were already L2 hits.  Do not expect it to scale with context.
     *
     * It is NOT bit-exact against the generic kernel: the online softmax folds
     * rows in a different order, so decode logits drift (top-1 held everywhere
     * sampled; top-10 reorders and a greedy stream diverges within a token or
     * two).  Each build stays perfectly deterministic run to run.  Prefill is
     * untouched — it already took this kernel. */
    /* There WAS a packed-comp clause here excluding multi-token non-descriptor
     * calls, on the grounds that "the f32 indexed kernel cannot read ATTN_PACK
     * rows there".  It can, and could all along: BOTH arms below take
     * comp_kv_pack and decode through attn_comp_pack_ld --
     * attention_indexed_mixed_heads8_online in its smem staging, and the
     * generic attention_indexed_mixed_kernel in the dot and the accumulation.
     * Decode has proved it every step since the format was unified: at
     * n_tokens == 1 the clause admitted packed rows and routed them to exactly
     * this heads8 kernel.
     *
     * Keeping it cost more than speed once all six prefill sites went packed:
     * the clause started REFUSING the shipped shape, so multi-token tier-off
     * fell through to the generic kernel -- 2.3x slower, and a different float
     * accumulation order, which moved the tier-off logits (max 29.474 ->
     * 28.028 on a 5530-token prompt).  Dropping it puts that shape back on the
     * heads8 kernel and the value back. */
    const int f16_idx_ok = pulsar_gpu_attention_prefill_reads_packed_comp() &&
                           head_dim == 512u && (n_head % 32u) == 0u && n_tokens > 1u;
    if (head_dim == 512 && top_k <= 512u && !no_indexed_heads8) {
        /* rb4 twopass has no pack support, so pack (and banked) always take the
         * online branch. */
        /* fp16 tensor-core tier for the indexed path -- it replaced the f32
         * kernel that was the largest single entry in the prefill map
         * (12.5%).  Banked descriptors and ATTN_PACK comp rows ride the fp16
         * tier too, each behind its own gate: bank isolation is proven
         * algebraically (tests/attn_f16_banked_test.cu -- a wrong-bank read
         * is plausible attention, not an error, so it needs a test that
         * cannot be fooled), and packed rows decode through the one shared
         * attn_comp_pack_ld.  What KEEPS the f32 online kernel: single-token
         * indexed decode (n_tokens == 1) and opted-out builds
         * (PULSAR_CUDA_ATTN_F16=0). */
        if (f16_idx_ok && topk_ptr) {
            static int announced = 0;
            if (!announced) {
                announced = 1;
                fprintf(stderr, "pulsar: indexed attention = fp16 tensor-core tier\n");
            }
            if (pulsar_gpu_attention_f16_indexed(
                    (float *)heads->ptr, sinks, (const pulsar_q_t *)q->ptr,
                    (const float *)raw_kv->ptr, (const float *)comp_kv->ptr,
                    (const int *)topk_ptr, n_tokens, pos0, n_raw, raw_cap,
                    raw_start, n_comp, top_k, window, ratio, n_head, head_dim, (const int *)positions_ptr,
                    (const int *)seq_id_ptr, comp_bank_ptrs_ptr,
                    comp_cap, descr ? n_banks : 1u, q_prep))
                return 1;
            fprintf(stderr, "pulsar: fp16 indexed attention FAILED; refusing to "
                            "fall through to the f32 kernel\n");
            return 0;
        }
        ATTN_Q_PREP_FALLBACK(q, n_tokens, pos0, positions);
        {   /* the two-pass rb4 alternative is gone; this is the only arm */
            dim3 grid(n_tokens, (n_head + 15u) / 16u, 1);
            attention_indexed_mixed_heads8_online_kernel<8, 16><<<grid, 512>>>((float *)heads->ptr,
                                                                               sinks,
                                                                               (const pulsar_q_t *)q->ptr,
                                                                               (const float *)raw_kv->ptr,
                                                                               (const float *)comp_kv->ptr,
                                                                               topk_ptr,
                                                                               n_tokens,
                                                                               pos0,
                                                                               n_raw,
                                                                               raw_cap,
                                                                               raw_start,
                                                                               n_comp,
                                                                               top_k,
                                                                               window,
                                                                               ratio,
                                                                               n_head,
                                                                               head_dim,
                                                                               positions_ptr,
                                                                               seq_id_ptr,
                                                                               comp_bank_ptrs_ptr,
                                                                               comp_cap,
                                                                               descr ? n_banks : 1u);
            return cuda_ok(cudaGetLastError(), "attention indexed online launch");
        }
    }
    ATTN_Q_PREP_FALLBACK(q, n_tokens, pos0, positions);
    dim3 grid(n_tokens, n_head, 1);
    attention_indexed_mixed_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                  sinks,
                                                  (const pulsar_q_t *)q->ptr,
                                                  (const float *)raw_kv->ptr,
                                                  (const float *)comp_kv->ptr,
                                                  topk_ptr,
                                                  n_tokens,
                                                  pos0,
                                                  n_raw,
                                                  raw_cap,
                                                  raw_start,
                                                  n_comp,
                                                  top_k,
                                                  window,
                                                  ratio,
                                                  n_head,
                                                  head_dim,
                                                  positions_ptr,
                                                  seq_id_ptr,
                                                  comp_bank_ptrs_ptr,
                                                  comp_cap,
                                                  descr ? n_banks : 1u);
    return cuda_ok(cudaGetLastError(), "attention indexed mixed launch");
}

static int attention_prefill_mixed_launch(
        pulsar_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *raw_kv,
        const pulsar_gpu_tensor *comp_kv,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim,
        const pulsar_gpu_q_prep *q_prep) {
    if (!heads || !q || !raw_kv || !model_map || n_tokens == 0 || ratio == 0 ||
        (n_comp != 0 && !comp_kv) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * PULSAR_Q_ELT_SIZE ||
        raw_kv->bytes < (uint64_t)n_tokens * PULSAR_ATTN_PACK_ROWBYTES(head_dim) ||
        /* Pack-aware, like the three sibling launches.  A guard that hard-codes
         * the f32 row stride demands 2048 B from a 584 B packed pool, fails,
         * and returns 0 -- which is "did not encode", not an error, so the
         * graph silently does not run.  ->bytes is just a number, so the
         * mismatch is type-legal and compiles clean. */
        head_dim <= PULSAR_ATTN_PACK_NROT ||
        ((head_dim - PULSAR_ATTN_PACK_NROT) % PULSAR_FP8_KV_BLOCK) != 0 ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp *
         (PULSAR_ATTN_PACK_ROWBYTES(head_dim))) ||
        false) {
        return 0;
    }
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    /* The fused online-softmax kernel can now carry the comp mask, so the
     * masked case no longer has to fall back to the unfused two-GEMM path that
     * materialises the whole score matrix. Gated while it is A/B'd against
     * that path; the fused kernel reassociates the softmax (online rescale vs
     * batch max/sum), so it is NOT bit-identical to the GEMM path. */
    /* PULSAR_ATTN_FUSED_COMP is gone -- no setter anywhere, so the masked case
     * never took the fused kernel and the A/B it gated never ran. */
    const int allow_fused = 1;
    /* One-shot: which branch actually serves this workload. Guessing at this
     * has been wrong twice; print it rather than infer it. */
    static int mixed_path_reported = 0;
    if (!mixed_path_reported) {
        mixed_path_reported = 1;
        const int takes_window = allow_fused && n_tokens > 1 && head_dim == 512;
        fprintf(stderr,
                "pulsar: ATTN-MIXED n_tokens=%u n_comp=%u -> %s\n",
                n_tokens, n_comp,
                takes_window ? "FUSED window kernel"
                             : (g_cublas_ready
                                    ? "unfused cuBLAS two-GEMM"
                                    : "generic per-token kernel"));
    }
    if (allow_fused && n_tokens > 1 && head_dim == 512) {
        /* fp16 tensor-core tier -- see the twin in the raw-window launcher.
         * This is the site that carries the traffic: the raw-window one runs
         * twice a prefill, this one runs per layer. */
        static const int use_f16_attn_mixed = pulsar_env_tier_on("PULSAR_CUDA_ATTN_F16");
        if (use_f16_attn_mixed && head_dim == 512u && (n_head % 16u) == 0u) {
            static int announced = 0;
            if (!announced) {
                announced = 1;
                fprintf(stderr, "pulsar: attention = fp16 tensor-core tier "
                                "(default; PULSAR_CUDA_ATTN_F16=0 opts out; operands rounded to fp16)\n");
            }
            if (pulsar_gpu_attention_f16_prefill(
                    (float *)heads->ptr, sinks, (const pulsar_q_t *)q->ptr,
                    (const float *)raw_kv->ptr,
                    n_comp ? (const float *)comp_kv->ptr : NULL,
                    n_tokens, n_comp, window, ratio, n_head, head_dim, q_prep))
                return 1;
            fprintf(stderr, "pulsar: fp16 attention tier FAILED; refusing to "
                            "fall through to the f32 kernel\n");
            return 0;
        }
    }
    ATTN_Q_PREP_FALLBACK(q, n_tokens, 0u, NULL);
    if (g_cublas_ready && n_tokens > 1 && head_dim == 512) {
        const uint32_t n_keys = n_tokens + n_comp;
        const uint64_t kv_count = (uint64_t)n_keys * head_dim;
        const uint64_t score_count = (uint64_t)n_head * n_tokens * n_keys;
        const uint64_t out_count = (uint64_t)n_head * n_tokens * head_dim;
        const uint64_t kv_bytes = kv_count * sizeof(float);
        const uint64_t score_bytes = score_count * sizeof(float);
        const uint64_t out_bytes = out_count * sizeof(float);
        /* The score GEMM's operands must agree in type, so a narrowed Q needs an
         * f16 KV operand.  Staging KV (n_keys*head_dim, stride 0 across heads)
         * rather than widening Q (n_tok*n_head*head_dim) is ~128x less work:
         * 4 MiB against 512 MiB at a 4096-token prefill.  It is also EXACT --
         * the packed KV is E4M3 with a power-of-two block scale, so every value
         * fits f16's mantissa and range with nothing to round.
         * The value GEMM below keeps the f32 copy: its B operand is post-softmax
         * P, and narrowing THAT would be a real precision change. */
        const uint64_t kv16_bytes = (sizeof(pulsar_q_t) == sizeof(float))
                                  ? 0ull : kv_count * sizeof(__half);
        const uint64_t tmp_bytes = ((kv_bytes + 255u) & ~255ull) +
                                   ((score_bytes + 255u) & ~255ull) + out_bytes +
                                   ((kv16_bytes + 255u) & ~255ull);
        cuda_arena ar;
        if (!cuda_arena_begin(&ar, tmp_bytes, "attention mixed cublas")) return 0;
        float *kv      = (float *)cuda_arena_take(&ar, kv_bytes, 256);
        float *scores  = (float *)cuda_arena_take(&ar, score_bytes, 256);
        float *out_tmp = (float *)cuda_arena_take(&ar, out_bytes, 256);
        if (!out_tmp) return 0;   /* take() latches: one check covers all three */
        __half *kv16 = NULL;
        if (kv16_bytes) {
            kv16 = (__half *)cuda_arena_take(&ar, kv16_bytes, 256);
            if (!kv16) return 0;
        }
        const void *q_eff = (const pulsar_q_t *)q->ptr;
        attention_prefill_pack_mixed_kv_kernel<<<(kv_count + 255) / 256, 256>>>(
                kv,
                (const float *)raw_kv->ptr,
                n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                n_tokens,
                n_comp,
                head_dim,
                NULL);
        if (!cuda_ok(cudaGetLastError(), "attention mixed kv pack launch")) return 0;
        const void *score_a = kv;
        if (kv16) {
            attention_prefill_pack_mixed_kv_kernel<<<(kv_count + 255) / 256, 256>>>(
                    NULL,
                    (const float *)raw_kv->ptr,
                    n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                    n_tokens,
                    n_comp,
                    head_dim,
                    kv16);
            if (!cuda_ok(cudaGetLastError(), "attention mixed kv f16 pack launch")) return 0;
            score_a = kv16;
        }
        const float alpha = rsqrtf((float)head_dim);
        const float beta = 0.0f;
        cublasStatus_t st;
            /* Ex rather than Sgemm so the operands follow the stored Q type;
             * accumulator and output stay f32. */
            st = cublasGemmStridedBatchedEx(g_cublas,
                CUBLAS_OP_T, CUBLAS_OP_N,
                (int)n_keys, (int)n_tokens, (int)head_dim,
                &alpha,
                score_a,      PULSAR_Q_CUDA_TYPE, (int)head_dim, (long long)0,
                q_eff,        PULSAR_Q_CUDA_TYPE, (int)(n_head * head_dim), (long long)head_dim,
                &beta,
                scores,       CUDA_R_32F,         (int)n_keys, (long long)n_keys * n_tokens,
                (int)n_head,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (!cublas_ok(st, "attention mixed score gemm")) return 0;
        dim3 sgrid(n_tokens, n_head, 1);
        attention_prefill_mixed_softmax_kernel<<<sgrid, 256>>>(
                scores,
                sinks,
                n_tokens,
                n_comp,
                window,
                ratio,
                n_keys,
                NULL);
        if (!cuda_ok(cudaGetLastError(), "attention mixed softmax launch")) return 0;
        const float one = 1.0f;
            st = cublasSgemmStridedBatched(g_cublas,
                                       CUBLAS_OP_N,
                                       CUBLAS_OP_N,
                                       (int)head_dim,
                                       (int)n_tokens,
                                       (int)n_keys,
                                       &one,
                                       kv,
                                       (int)head_dim,
                                       0,
                                       scores,
                                       (int)n_keys,
                                       (long long)n_keys * n_tokens,
                                       &beta,
                                       out_tmp,
                                       (int)head_dim,
                                       (long long)head_dim * n_tokens,
                                       (int)n_head);
        if (!cublas_ok(st, "attention mixed value gemm")) return 0;
        uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
        attention_prefill_unpack_heads_kernel<<<(n + 255) / 256, 256>>>((float *)heads->ptr,
                                                                        out_tmp,
                                                                        n_tokens,
                                                                        n_head,
                                                                        head_dim);
        return cuda_ok(cudaGetLastError(), "attention mixed unpack launch");
    }
    dim3 grid(n_tokens, n_head, 1);
    attention_prefill_mixed_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                  sinks,
                                                  (const pulsar_q_t *)q->ptr,
                                                  (const float *)raw_kv->ptr,
                                                  n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                  n_tokens, n_comp, window, ratio,
                                                  n_head, head_dim);
    return cuda_ok(cudaGetLastError(), "attention prefill mixed launch");
}

int pulsar_gpu_attention_prefill_static_mixed_heads_tensor(
        pulsar_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *raw_kv,
        const pulsar_gpu_tensor *comp_kv,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim,
        const pulsar_gpu_q_prep *q_prep) {
    return attention_prefill_mixed_launch(heads, model_map, model_size, sinks_offset,
                                       q, raw_kv, comp_kv, n_tokens,
                                       n_comp, window, ratio, n_head, head_dim,
                                       q_prep);
}


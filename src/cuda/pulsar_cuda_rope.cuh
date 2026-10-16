/* Shared tail-rope math. The SINGLE authority for the YaRN-corrected rotation
 * every tail-rope consumer applies -- head_rms_norm_rope_tail_kernel and
 * rope_tail_kernel (pulsar_cuda_norm_kv.cu) and the fused Q-load in
 * attn_f16_kernel (pulsar_cuda_attn_f16.cu). Three kernels computing this
 * independently is how bit-exactness dies; they all call these.
 *
 * ⚠ THE "they all call these" CLAIM WAS FALSE FROM 2026-08-21 TO 2026-08-24.
 * The header shipped and attn_f16_kernel used it, but pulsar_cuda_norm_kv.cu
 * never included it and kept TWO transcribed copies plus its own ramp helper --
 * and one of those copies also described itself as "the SINGLE authority".
 * Three transcriptions, two of them claiming to be the only one. L074 closed
 * that on 2026-08-24; the claim above is now true, and `grep -L
 * pulsar_cuda_rope.cuh $(grep -rl rope_pair_rotate src/cuda)` is how to keep
 * it true.
 *
 * Every operation and its order is preserved from the original kernel bodies
 * verbatim: a prefill gate depth compares full-vocab logits byte-for-byte, so
 * even a reassociated multiply here is a red gate. */
#pragma once

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ static inline float rope_yarn_ramp_core_dev(float low, float high, int i0) {
    float y = ((float)(i0 / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

/* The YaRN correction window, a pure function of the rope shape. Callers hoist
 * it out of per-pair loops exactly as the original kernels did. */
__device__ static inline void rope_corr_dims_dev(
        uint32_t n_rot, uint32_t n_ctx_orig, float freq_base,
        float beta_fast, float beta_slow, float *corr0, float *corr1) {
    float denom = 2.0f * logf(freq_base);
    float c0 = floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
    float c1 = ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
    *corr0 = fmaxf(0.0f, c0);
    *corr1 = fminf((float)(n_rot - 1), c1);
}

/* Rotate one tail pair (x0, x1) at tail index i (even). corr0/corr1 come from
 * rope_corr_dims_dev when ext_factor != 0 (they are ignored otherwise). */
__device__ static inline void rope_pair_rotate_core_dev(
        float x0, float x1, uint32_t i, uint32_t n_rot, uint32_t rope_pos,
        int inverse, float freq_base, float freq_scale,
        float ext_factor, float attn_factor,
        float corr0, float corr1, float *r0, float *r1) {
    float theta_extrap = (float)rope_pos * powf(freq_base, -((float)i) / (float)n_rot);
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = attn_factor;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp_core_dev(corr0, corr1, (int)i) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    float c = cosf(theta) * mscale;
    float s = sinf(theta) * mscale;
    if (inverse) s = -s;
    *r0 = x0 * c - x1 * s;
    *r1 = x0 * s + x1 * c;
}

/* The NeoX (half-split) rotation in the HF / vLLM bf16 arithmetic (L298: the block drafter's q and k; a Qwen3-style
 * layer).  Pair p of a head of d dims is (x[p], x[p + d/2]); its frequency is the f32 inv_freq vLLM caches,
 * 1 / base^(2p / d), its angle pos * inv_freq in f32; cos and sin are rounded to bf16 (the cache vLLM casts to the
 * model dtype, HF's .to(x.dtype)), and each product and the difference / sum is one bf16 rounding (c10::BFloat16
 * arithmetic, HF's bf16 tensor ops).  x0 / x1 are bf16 values held in f32; r0 / r1 come back the same.
 *
 * The engine builds with --use_fast_math, which turns powf / cosf / sinf into the __powf / __cosf / __sinf
 * approximations and 1.0f / x into an approximate reciprocal: at an angle near 1e5 rad that moved ~6% of the stored
 * K elements by a bf16 ulp or more (tests/gqa_attn_gate, L298).  So pow, cos and sin are evaluated in double (which
 * fast-math leaves alone) and rounded once to f32 -- the f32 value torch's / libm's f32 functions return -- and the
 * division and the product are the IEEE intrinsics. */
__device__ static inline float rope_neox_bf16_round_dev(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ static inline void rope_neox_bf16_pair_dev(float x0, float x1, uint32_t p, uint32_t d, uint32_t pos,
                                                      float base, float *r0, float *r1) {
    const float e = __fdiv_rn((float)(2u * p), (float)d);
    const float inv = __fdiv_rn(1.0f, (float)pow((double)base, (double)e));
    const double ang = (double)__fmul_rn((float)pos, inv);
    const float c = rope_neox_bf16_round_dev((float)cos(ang)), s = rope_neox_bf16_round_dev((float)sin(ang));
    *r0 = rope_neox_bf16_round_dev(__fsub_rn(rope_neox_bf16_round_dev(__fmul_rn(x0, c)), rope_neox_bf16_round_dev(__fmul_rn(x1, s))));
    *r1 = rope_neox_bf16_round_dev(__fadd_rn(rope_neox_bf16_round_dev(__fmul_rn(x1, c)), rope_neox_bf16_round_dev(__fmul_rn(x0, s))));
}

/** @file
 *  Shared tail-RoPE math: the single authority for the YaRN-corrected rotation. */
/* Shared tail-rope math. The SINGLE authority for the YaRN-corrected rotation
 * (the YaRN formulas themselves are lib/pulsar_yarn.h's, shared with Qwen's QSA
 * rope since L280) every tail-rope consumer applies -- head_rms_norm_rope_tail_kernel and
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

#include "lib/pulsar_yarn.h"   /* the YaRN math itself: ONE authority for every family (L280) */

/* The YaRN correction window, a pure function of the rope shape. Callers hoist
 * it out of per-pair loops exactly as the original kernels did. */
/// @param n_rot      rotated dims
/// @param n_ctx_orig the model's original training context
/// @param freq_base  rope base
/// @param beta_fast  YaRN beta_fast
/// @param beta_slow  YaRN beta_slow
/// @param corr0      out: the window's low dim
/// @param corr1      out: the window's high dim
__device__ static inline void rope_corr_dims_dev(
        uint32_t n_rot, uint32_t n_ctx_orig, float freq_base,
        float beta_fast, float beta_slow, float *corr0, float *corr1) {
    pulsar_yarn_corr_range<float>(n_rot, n_ctx_orig, freq_base, beta_fast, beta_slow, corr0, corr1);
}

/* Rotate one tail pair (x0, x1) at tail index i (even). corr0/corr1 come from
 * rope_corr_dims_dev when ext_factor != 0 (they are ignored otherwise). */
/// @param x0          the pair's first element
/// @param x1          the pair's second element
/// @param i           tail index of the pair (even)
/// @param n_rot       rotated dims
/// @param rope_pos    the token's position
/// @param inverse     nonzero: rotate by -theta
/// @param freq_base   rope base
/// @param freq_scale  YaRN frequency scale (1 = none)
/// @param ext_factor  YaRN extrapolation mix (0 = none: corr0 / corr1 ignored)
/// @param attn_factor magnitude scale
/// @param corr0       YaRN window low dim (rope_corr_dims_dev)
/// @param corr1       YaRN window high dim
/// @param r0          out: rotated first element
/// @param r1          out: rotated second element
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
        float ramp_mix = pulsar_yarn_extrap_weight<float>(corr0, corr1, (float)((int)i / 2)) * ext_factor;
        theta = pulsar_yarn_blend<float>(theta_interp, theta_extrap, ramp_mix);
        mscale *= pulsar_yarn_mscale<float>(1.0f / freq_scale);
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
/// @param x the value
/// @return x rounded to bf16 (nearest even), as f32
__device__ static inline float rope_neox_bf16_round_dev(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
/* The rotation at a GIVEN inverse frequency `inv` (L321: a scaled rope -- YaRN -- hands its per-pair table here;
 * src/lib/pulsar_yarn.h builds it on the host). */
/// @param x0  pair element p (a bf16 value in f32)
/// @param x1  pair element p + d/2
/// @param inv the pair's inverse frequency
/// @param pos the token's position
/// @param r0  out: rotated x0 (bf16 value in f32)
/// @param r1  out: rotated x1
__device__ static inline void rope_neox_bf16_pair_inv_dev(float x0, float x1, float inv, uint32_t pos, float *r0, float *r1) {
    const double ang = (double)__fmul_rn((float)pos, inv);
    const float c = rope_neox_bf16_round_dev((float)cos(ang)), s = rope_neox_bf16_round_dev((float)sin(ang));
    *r0 = rope_neox_bf16_round_dev(__fsub_rn(rope_neox_bf16_round_dev(__fmul_rn(x0, c)), rope_neox_bf16_round_dev(__fmul_rn(x1, s))));
    *r1 = rope_neox_bf16_round_dev(__fadd_rn(rope_neox_bf16_round_dev(__fmul_rn(x1, c)), rope_neox_bf16_round_dev(__fmul_rn(x0, s))));
}
/// The NeoX rotation of pair p at inverse frequency 1 / base^(2p / d) (see above).
/// @param x0   pair element p (a bf16 value in f32)
/// @param x1   pair element p + d/2
/// @param p    pair index, 0 .. d/2 - 1
/// @param d    head dims
/// @param pos  the token's position
/// @param base rope base
/// @param r0   out: rotated x0
/// @param r1   out: rotated x1
__device__ static inline void rope_neox_bf16_pair_dev(float x0, float x1, uint32_t p, uint32_t d, uint32_t pos,
                                                      float base, float *r0, float *r1) {
    const float e = __fdiv_rn((float)(2u * p), (float)d);
    const float inv = __fdiv_rn(1.0f, (float)pow((double)base, (double)e));
    rope_neox_bf16_pair_inv_dev(x0, x1, inv, pos, r0, r1);
}

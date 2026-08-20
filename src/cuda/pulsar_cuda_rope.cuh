/* Shared tail-rope math. The SINGLE authority for the YaRN-corrected rotation
 * every tail-rope consumer applies -- head_rms_norm_rope_tail_kernel and
 * rope_tail_kernel (pulsar_cuda_norm_kv.cu) and the fused Q-load in
 * attn_f16_kernel (pulsar_cuda_attn_f16.cu). Three kernels computing this
 * independently is how bit-exactness dies; they all call these.
 *
 * Every operation and its order is preserved from the original kernel bodies
 * verbatim: a prefill gate depth compares full-vocab logits byte-for-byte, so
 * even a reassociated multiply here is a red gate. */
#pragma once

#include <math.h>

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

/* YaRN rope scaling (Peng et al. 2023, arXiv 2309.00071): the ONE authority for its math, for every family (L280).
 *
 * A YaRN rope keeps the native rope's high-frequency pairs (they complete many rotations inside the original
 * context, so extrapolating them is safe), interpolates the low-frequency ones by the scaling factor (they never
 * completed a rotation, so positions past the original context would be new angles), blends the pairs between on a
 * linear ramp, and scales cos / sin by the attention factor 1 + 0.1 ln(factor).  Two reference formulations exist
 * and they are the same math:
 *
 *   transformers `_compute_yarn_parameters`  blends the inverse FREQUENCIES once, and the angle is pos * inv_freq;
 *                                            corrections, ramp and attention factor in Python double, frequencies in
 *                                            torch f32
 *   ggml `rope_yarn`                         blends the ANGLES per pair (theta_interp, theta_extrap), all f32
 *
 * DeepSeek's tail rope (pulsar_cuda_rope.cuh) is the ggml form, on the device, f32 under --use_fast_math, and its
 * bytes are pinned by the prefill / decode byte gates.  Qwen's QSA rope (pulsar_qsa_rope_init) is the transformers
 * form, evaluated once per load on the host in double and rounded once to f32 (the L298 lesson: a rope frequency is
 * never a fast-math __powf).  Both call these helpers with their own arithmetic type F; each helper is written as
 * the expression both references share, so DeepSeek's instantiation compiles to the instructions it had before
 * these were shared.
 *
 * The weight helpers return the EXTRAPOLATION weight (transformers' inv_freq_extrapolation_factor, ggml's ramp_mix
 * at ext_factor 1): 1 = keep the native frequency, 0 = fully interpolated. */
#pragma once

#include <math.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#if defined(__CUDACC__)
#define PULSAR_YARN_FN __host__ __device__ static inline
#else
#define PULSAR_YARN_FN static inline
#endif

/** The pair index at which a rope of `n_dims` rotary dims and base `base` completes `n_rotations` rotations over
 *  `n_ctx_orig` positions (transformers' find_correction_dim, ggml's ggml_rope_yarn_corr_dim). */
template <typename F>
PULSAR_YARN_FN F pulsar_yarn_corr_dim(uint32_t n_dims, uint32_t n_ctx_orig, F base, F n_rotations) {
    return (F)n_dims * log((F)n_ctx_orig / (n_rotations * (F)2 * (F)M_PI)) / ((F)2 * log(base));
}

/** The ramp's pair range [low, high]: beta_fast rotations bound the extrapolated pairs, beta_slow the interpolated
 *  ones, truncated outward to whole pairs and clamped to the rotary dims (transformers' find_correction_range with
 *  truncate, ggml's ggml_rope_yarn_corr_dims). */
template <typename F>
PULSAR_YARN_FN void pulsar_yarn_corr_range(uint32_t n_dims, uint32_t n_ctx_orig, F base, F beta_fast, F beta_slow,
                                           F *low, F *high) {
    const F lo = floor(pulsar_yarn_corr_dim<F>(n_dims, n_ctx_orig, base, beta_fast));
    const F hi = ceil(pulsar_yarn_corr_dim<F>(n_dims, n_ctx_orig, base, beta_slow));
    *low = fmax((F)0, lo);
    *high = fmin((F)(n_dims - 1u), hi);
}

/** Pair `pair`'s extrapolation weight: 1 below `low`, 0 above `high`, linear between (1 - transformers'
 *  linear_ramp_factor, ggml's rope_yarn_ramp).  A degenerate range (high <= low) is a step at `low`. */
template <typename F>
PULSAR_YARN_FN F pulsar_yarn_extrap_weight(F low, F high, F pair) {
    const F y = (pair - low) / fmax((F)0.001, high - low);
    return (F)1 - fmin((F)1, fmax((F)0, y));
}

/** The blend: `interp` (scaled by 1 / factor) and `extrap` (native) mixed by the extrapolation weight `w` --
 *  frequencies in transformers' form, angles in ggml's. */
template <typename F>
PULSAR_YARN_FN F pulsar_yarn_blend(F interp, F extrap, F w) {
    return interp * ((F)1 - w) + extrap * w;
}

/** The attention factor on cos and sin: 1 + 0.1 ln(factor) (transformers' get_mscale at mscale 1; ggml's
 *  1 + 0.1 logf(1 / freq_scale)).  Defined for factor > 1, the only factor a YaRN rope takes. */
template <typename F>
PULSAR_YARN_FN F pulsar_yarn_mscale(F factor) {
    return (F)1 + (F)0.1 * log(factor);
}

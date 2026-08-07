#include "pulsar_engine_internal.h"


























/* Dense layers and compressed layers use different RoPE bases. */
float layer_rope_freq_base(uint32_t il) {
    return pulsar_layer_compress_ratio(il) != 0 && PULSAR_COMPRESS_ROPE_FREQ_BASE > 0.0f
        ? PULSAR_COMPRESS_ROPE_FREQ_BASE
        : PULSAR_ROPE_FREQ_BASE;
}



float layer_rope_freq_scale(uint32_t il) {
    if (pulsar_layer_compress_ratio(il) == 0 || PULSAR_ROPE_SCALE_FACTOR <= 0.0f) {
        return 1.0f;
    }
    return 1.0f / PULSAR_ROPE_SCALE_FACTOR;
}












float sigmoid_stable(float x) {
    if (x >= 0.0f) {
        const float e = expf(-x);
        return 1.0f / (1.0f + e);
    } else {
        const float e = expf(x);
        return e / (1.0f + e);
    }
}


















/* =========================================================================
 * Mixture-of-Experts FFN.
 * =========================================================================
 *
 * This is the FFN half of each layer.  It includes the shared expert, routed
 * expert selection, IQ2_XXS gate/up projections, SwiGLU, Q2_K down projection,
 * and the HC post step that returns the result to four-stream state.
 */



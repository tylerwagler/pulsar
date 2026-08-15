#include "pulsar_engine_internal.h"



uint32_t pulsar_default_raw_cap(uint32_t ctx_size) {
    uint32_t raw_cap = PULSAR_N_SWA;
    if (raw_cap > ctx_size) raw_cap = ctx_size;
    if (raw_cap == 0) raw_cap = 1;
    return raw_cap;
}



uint32_t pulsar_prefill_cap_for_prompt(int prompt_len,
                                           uint32_t requested_chunk) {
    if (prompt_len <= 0) return 1;
    uint32_t cap = (uint32_t)prompt_len;

    if (requested_chunk != 0) {
        cap = requested_chunk;
    } else {
        const char *env = getenv("PULSAR_CUDA_PREFILL_CHUNK");
        if (env && env[0]) {
            char *endp = NULL;
            const long v = strtol(env, &endp, 10);
            if (endp != env) {
                if (v <= 0) return cap;
                cap = (uint32_t)v;
            }
        } else if (prompt_len > 4096) {
            cap = 4096u;
        }
    }

    if (cap == 0) cap = 1;
    if (cap > (uint32_t)prompt_len) cap = (uint32_t)prompt_len;
    return cap;
}















int sample_argmax(const float *logits, uint32_t n_vocab);



/* =========================================================================
 * GPU Reference Comparison Helpers.
 * =========================================================================
 *
 * These small scalar helpers are used only by diagnostics that compare the C
 * reference path with the GPU executor.
 */

float max_abs_diff(const float *a, const float *b, uint64_t n) {
    float max_diff = 0.0f;
    for (uint64_t i = 0; i < n; i++) {
        const float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}


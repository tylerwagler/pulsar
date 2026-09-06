#include "pulsar_engine_internal.h"



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
        } else if ((uint32_t)prompt_len > PULSAR_PREFILL_CHUNK_DEFAULT) {
            cap = PULSAR_PREFILL_CHUNK_DEFAULT;
        }
    }

    if (cap == 0) cap = 1;
    if (cap > (uint32_t)prompt_len) cap = (uint32_t)prompt_len;
    return cap;
}


















/* =========================================================================
 * GPU Reference Comparison Helpers.
 * =========================================================================
 *
 * These small scalar helpers are used only by diagnostics that compare the C
 * reference path with the GPU executor.
 */



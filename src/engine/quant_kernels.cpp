#include "pulsar_engine_internal.h"













































/* =========================================================================
 * Fixed Weight Binding and Model Validation.
 * =========================================================================
 *
 * The GGUF tensor directory is converted into a DS4-specific pointer table.
 * After this section, the rest of the program addresses tensors by semantic
 * fields such as layer->attn_q_a or layer->ffn_gate_exps rather than by string
 * lookup.  Shape validation is intentionally strict.
 */

uint32_t required_u32(const pulsar_model *m, const char *key) {
    uint32_t v = 0;
    if (!model_get_u32(m, key, &v)) {
        fprintf(stderr, "pulsar: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return v;
}


/* tensor_bind.cpp -- L272 P4a: the mechanics every family's weight binder shares.
 *
 * Which tensors a family's layers hold, and which formats each role admits, is the family's knowledge
 * (weights.cpp for DeepSeek: per-mode attention ownership, hash-routed layers, the drafter and the
 * vision tower; family_qwen.cpp / family_qwen_s4.cpp for Qwen).  Finding a tensor, checking its dims
 * and admitting its format are not: these three report the same way for every family and return the
 * verdict, and each binder keeps its own failure policy (DeepSeek stops at the first, Qwen collects
 * every refusal before it refuses the load). */
#include "pulsar_engine_internal.h"

pulsar_tensor *pulsar_tensor_bind(const pulsar_model *m, const char *owner, const char *name) {
    pulsar_tensor *t = model_find_tensor(m, name);
    if (!t) fprintf(stderr, "pulsar: %s: required tensor %s is missing\n", owner, name);
    return t;
}

bool pulsar_tensor_dims(const pulsar_tensor *t, const char *owner, uint32_t nd, uint64_t d0, uint64_t d1,
                        uint64_t d2) {
    const uint64_t want[3] = {d0, d1, d2};
    bool ok = t->ndim == nd;
    for (uint32_t i = 0; ok && i < nd; i++) ok = t->dim[i] == want[i];
    if (ok) return true;
    fprintf(stderr, "pulsar: %s: tensor %.*s has ne [", owner, (int)t->name.len, t->name.ptr);
    for (uint32_t i = 0; i < t->ndim; i++) fprintf(stderr, "%s%llu", i ? ", " : "", (unsigned long long)t->dim[i]);
    fprintf(stderr, "], want [");
    for (uint32_t i = 0; i < nd; i++) fprintf(stderr, "%s%llu", i ? ", " : "", (unsigned long long)want[i]);
    fprintf(stderr, "]\n");
    return false;
}

bool pulsar_tensor_admit(const pulsar_tensor *t, const char *owner, bool ok, const char *want) {
    if (ok) return true;
    fprintf(stderr, "pulsar: %s: tensor %.*s is %s; the op that reads it takes %s -- refusing\n", owner,
            (int)t->name.len, t->name.ptr, tensor_type_name(t->type), want);
    return false;
}

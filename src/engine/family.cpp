/* family.cpp -- the family registry and the capability check (L251 S1).
 * See family.h for the contract. */
#include "pulsar_engine_internal.h"

const char *pulsar_layer_kind_name(pulsar_layer_kind k) {
    switch (k) {
    case PULSAR_LAYER_NONE:      return "none";
    case PULSAR_LAYER_DS4_BLOCK: return "ds4_block";
    case PULSAR_LAYER_QWEN_GDN:  return "qwen_gdn";
    case PULSAR_LAYER_QWEN_QSA:  return "qwen_qsa";
    }
    return "unknown";
}

uint32_t pulsar_layer_plan_count(const pulsar_layer_plan *p, pulsar_layer_kind k) {
    uint32_t n = 0;
    for (uint32_t il = 0; il < p->n_layer; il++) n += p->kind[il] == k;
    return n;
}

/* Every family the engine serves.  The ONE place an architecture string is
 * mapped to code. */
static const pulsar_family *const k_families[] = {
    &PULSAR_FAMILY_DEEPSEEK4,
    &PULSAR_FAMILY_QWEN4_EXP,
};

const pulsar_family *pulsar_family_for_model(const pulsar_model *m) {
    pulsar_str arch = {NULL, 0};
    if (!model_get_string(m, "general.architecture", &arch)) {
        fprintf(stderr, "pulsar: the artifact has no general.architecture key; "
                        "no model family can claim it -- refusing\n");
        return NULL;
    }
    for (size_t i = 0; i < sizeof(k_families) / sizeof(k_families[0]); i++) {
        const char *a = k_families[i]->arch;
        if (strlen(a) == arch.len && memcmp(a, arch.ptr, arch.len) == 0) return k_families[i];
    }
    fprintf(stderr, "pulsar: general.architecture '%.*s' is not a family this engine serves (",
            (int)arch.len, arch.ptr);
    for (size_t i = 0; i < sizeof(k_families) / sizeof(k_families[0]); i++)
        fprintf(stderr, "%s'%s'", i ? ", " : "", k_families[i]->arch);
    fprintf(stderr, ") -- refusing\n");
    return NULL;
}

bool pulsar_family_require(const pulsar_engine *e, uint32_t cap, const char *op) {
    if (!e || (e->family->caps & cap) == cap) return true;
    /* Once per op name (rule 9): a caller that keeps asking gets its refusal
     * value every time, and the log gets the reason once.  The op names are
     * string literals, so their addresses identify them. */
    static const char *said[64];
    static uint32_t n_said = 0;
    for (uint32_t i = 0; i < n_said; i++)
        if (said[i] == op) return false;
    if (n_said < sizeof(said) / sizeof(said[0])) said[n_said++] = op;
    fprintf(stderr, "pulsar: the %s family does not implement %s -- refusing\n",
            e->family->name, op);
    return false;
}

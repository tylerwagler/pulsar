/* weight_format.cpp -- L272 P4: the weight formats are the core's, not a family's.
 *
 * Tyler: "the quantization type should not be tied to the family. It's very possible that I'll want to run a
 * Deepseek exl3 later or an NVFP4 Qwen."  A family says what ROLE each tensor plays (a dense linear, a routed
 * expert's gate / up / down, a shared expert inside the MoE launcher) and which ACTIVATION its forward emits at
 * that site (f32, bf16 rows, or the E4M3 MX slot).  Whether a stored format can serve that (role, activation) is
 * this table's answer -- one per kernel the engine has (research/family-plugin-audit-2026-10-06/09):
 *
 *   dense            f32 act: F32, BF16 (the plain cuBLAS arms)   E4M3: MXFP8_LT (cuBLASLt)
 *                    bf16 act: MXFP8_LT (split-K GEMV / MMA), EXL3 at a dense-arm rate
 *   expert gate/up   E4M3: CUTLASS MXFP4, IQ2 MMQ, EXL3 at a pair-arm rate        bf16: EXL3 pair-arm
 *   expert down      E4M3: CUTLASS MXFP4, IQ2 MMQ, EXL3 at a down-arm rate        bf16: EXL3 down-arm
 *   fused gate_up    bf16: EXL3 at a fused-arm rate
 *   shared expert    bf16: EXL3 at a dense-arm rate (inside the bf16 MoE launcher)
 *
 * A combination with no kernel refuses by name at load -- never a converted activation (rule 3: producers emit,
 * consumers never convert).  Filling one (an E4M3-activation EXL3 dense arm for a DeepSeek EXL3 artifact, a bf16
 * CUTLASS MXFP4 MoE for a Qwen MXFP4 artifact, NVFP4 weights) is a kernel plus a line here. */
#include "pulsar_engine_internal.h"
#include "exl3_trellis.h"

#include <string>

static bool exl3_arm(uint32_t type, int arm) {
    const int k2 = exl3_type_k2(type);
    return k2 != 0 && exl3_arm_has_rate(arm, k2);
}

bool pulsar_format_serves(uint32_t type, pulsar_weight_role role, pulsar_act_format act) {
    switch (role) {
    case PULSAR_ROLE_DENSE:
        if (act == PULSAR_ACT_F32) return type == PULSAR_TENSOR_F32 || type == PULSAR_TENSOR_BF16;
        if (act == PULSAR_ACT_E4M3) return type == PULSAR_TENSOR_MXFP8_LT;
        return type == PULSAR_TENSOR_MXFP8_LT || exl3_arm(type, EXL3_ARM_DENSE);
    case PULSAR_ROLE_EXPERT_GATE_UP:
        if (act == PULSAR_ACT_E4M3)
            return type == PULSAR_TENSOR_CUTLASS_MXFP4 || type == PULSAR_TENSOR_IQ2_XXS_MMQ_K || exl3_arm(type, EXL3_ARM_PAIR);
        return act == PULSAR_ACT_BF16 && exl3_arm(type, EXL3_ARM_PAIR);
    case PULSAR_ROLE_EXPERT_DOWN:
        if (act == PULSAR_ACT_E4M3)
            return type == PULSAR_TENSOR_CUTLASS_MXFP4 || type == PULSAR_TENSOR_IQ2_XXS_MMQ_K || exl3_arm(type, EXL3_ARM_DOWN);
        return act == PULSAR_ACT_BF16 && exl3_arm(type, EXL3_ARM_DOWN);
    case PULSAR_ROLE_EXPERT_GATE_UP_FUSED:
        return act == PULSAR_ACT_BF16 && exl3_arm(type, EXL3_ARM_GATE_UP_FUSED);
    case PULSAR_ROLE_SHARED_EXPERT:
        return act == PULSAR_ACT_BF16 && exl3_arm(type, EXL3_ARM_DENSE);
    }
    return false;
}

static const char *const k_role_name[] = {"a dense linear", "a routed expert's gate / up", "a routed expert's down",
                                          "a fused routed gate_up", "a shared expert"};
static const char *const k_act_name[] = {"f32", "bf16", "E4M3-MX"};

bool pulsar_tensor_admit_role(const pulsar_tensor *t, const char *owner, pulsar_weight_role role, uint32_t acts) {
    for (int a = 0; a < PULSAR_ACT_COUNT; a++)
        if ((acts >> a & 1u) && pulsar_format_serves(t->type, role, (pulsar_act_format)a)) return true;
    /* the refusal names what WOULD serve: every known format with a kernel for this role at these activations */
    std::string want, act_list;
    for (int a = 0; a < PULSAR_ACT_COUNT; a++)
        if (acts >> a & 1u) act_list += std::string(act_list.empty() ? "" : " or ") + k_act_name[a];
    for (uint32_t type = 0; type < 256u; type++) {
        if (!strcmp(tensor_type_name(type), "unknown")) continue;
        bool any = false;
        for (int a = 0; a < PULSAR_ACT_COUNT; a++) any |= (acts >> a & 1u) && pulsar_format_serves(type, role, (pulsar_act_format)a);
        if (any) want += std::string(want.empty() ? "" : ", ") + tensor_type_name(type);
    }
    const std::string what = std::string(k_role_name[role]) + " at " + act_list + " activations: " +
                             (want.empty() ? std::string("(no format has a kernel)") : want);
    return pulsar_tensor_admit(t, owner, false, what.c_str());
}

bool pulsar_format_moe_combo(const pulsar_tensor *gate, const pulsar_tensor *up, const pulsar_tensor *down,
                             pulsar_act_format act, const char *owner) {
    /* the MoE launchers' own constraints: a gate / up pair is one format (the pair kernels read both with one
     * decode); at E4M3 activations an EXL3 side runs its own arm on both projections, so it never pairs with a
     * CUTLASS MXFP4 / IQ2 side (the rates may differ: each side decodes by its own type) */
    if (up && up->type != gate->type)
        return pulsar_tensor_admit(up, owner, false, "the gate stack's format (a gate / up pair is one format)");
    if (act == PULSAR_ACT_E4M3 && (exl3_type_k2(gate->type) != 0) != (exl3_type_k2(down->type) != 0))
        return pulsar_tensor_admit(down, owner, false,
                                   "the gate side's family of formats (an EXL3 side pairs only with an EXL3 side)");
    return true;
}

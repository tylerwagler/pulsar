/* moe.cpp -- L272 P4c: the routed MoE's one front door, for every family.
 *
 * The routed part of an MoE block -- the selected experts' gate / up, the SwiGLU, down, the weighted sum -- is a
 * function of the expert stacks, the routing (selected ids and weights) and the block input.  The router and a
 * shared expert are the family's architecture and stay outside.  The family names its stacks and hands the
 * activation as it produced it; the arm is the core's choice by (gate / up format, down format, activation),
 * from the format registry admission also reads (pulsar_format_serves / pulsar_format_moe_combo):
 *
 *   the MX SLOT (E4M3 armed in the backend's cache for the f32 block input -- DeepSeek; pulsar_moe_routed_slot):
 *       CUTLASS MXFP4, IQ2 MMQ, EXL3, and the mixed CUTLASS / MMQ shapes -> the backend's routed dispatcher
 *   raw bf16 ROWS (Qwen; pulsar_moe_routed_rows):
 *       EXL3 fused gate_up + down, or an EXL3 gate + up pair + down -> pulsar_rows_moe_routed_launch
 *
 * What a tensor-parallel rank reads is the rank's plan's (tp_slice.cpp), not the family's: a stack the plan
 * halved (expert tensor-parallel, MXFP4) reads the halves registered under the engine key; a stack the plan
 * ranged (expert parallel) reads that range of whole experts.  Every arm is today's launcher, unchanged. */
#include "pulsar_engine_internal.h"
#include "exl3_trellis.h"
#include "cuda/pulsar_cuda_qwen.h"

pulsar_moe_arm pulsar_moe_arm_for(const pulsar_tensor *gate, const pulsar_tensor *up, const pulsar_tensor *down,
                                  pulsar_act_format act) {
    if (!gate || !down) return PULSAR_MOE_ARM_NONE;
    if (up && up->type != gate->type) return PULSAR_MOE_ARM_NONE;   /* a gate / up pair is one format */
    if (act == PULSAR_ACT_E4M3) {
        /* at E4M3 an EXL3 side runs its own arm on both projections: it never pairs with a CUTLASS / MMQ side */
        const bool ok = up && pulsar_format_serves(gate->type, PULSAR_ROLE_EXPERT_GATE_UP, act) &&
                        pulsar_format_serves(down->type, PULSAR_ROLE_EXPERT_DOWN, act) &&
                        (exl3_type_k2(gate->type) != 0) == (exl3_type_k2(down->type) != 0);
        return ok ? PULSAR_MOE_ARM_SLOT : PULSAR_MOE_ARM_NONE;
    }
    if (act == PULSAR_ACT_BF16 && pulsar_format_serves(down->type, PULSAR_ROLE_EXPERT_DOWN, act)) {
        if (!up && pulsar_format_serves(gate->type, PULSAR_ROLE_EXPERT_GATE_UP_FUSED, act)) return PULSAR_MOE_ARM_ROWS_FUSED;
        if (up && pulsar_format_serves(gate->type, PULSAR_ROLE_EXPERT_GATE_UP, act)) return PULSAR_MOE_ARM_ROWS_PAIR;
    }
    return PULSAR_MOE_ARM_NONE;
}

static void say_no_arm(const pulsar_tensor *gate, const pulsar_tensor *down, const char *against) {
    fprintf(stderr, "pulsar: routed MoE: %.*s (%s) with %.*s (%s) has no arm against %s -- refusing\n",
            (int)gate->name.len, gate->name.ptr, tensor_type_name(gate->type), (int)down->name.len, down->name.ptr,
            tensor_type_name(down->type), against);
}

bool pulsar_moe_routed_slot(const pulsar_moe_slot_call *c) {
    const pulsar_tensor *G = c->gate, *U = c->up, *D = c->down;
    if (pulsar_moe_arm_for(G, U, D, PULSAR_ACT_E4M3) != PULSAR_MOE_ARM_SLOT) {
        say_no_arm(G, D, "the MX slot (E4M3)");
        return false;
    }
    const uint64_t in = G->dim[0], out = D->dim[1];
    uint64_t mid = D->dim[0];
    const void *map = tensor_map_base(c->m, G);
    uint64_t size = tensor_map_size(c->m, G);
    uint64_t gate_off = G->abs_offset, up_off = U->abs_offset, down_off = D->abs_offset;
    uint32_t split = 0;
    /* the rank's plan halved the stacks (expert tensor-parallel): every expert at the half intermediate width,
     * from the compact half stacks built at open, resolved under the engine key -- a partial the FFN exchange sums */
    pulsar_tp_op op = PULSAR_TP_OP_NONE;
    uint64_t lo = 0, hi = 0;
    const void *key = NULL;
    if (pulsar_tp_slice_of(c->m, G, &op, &lo, &hi, &key)) {
        if (op != PULSAR_TP_OP_MXFP4_HALF) {
            fprintf(stderr, "pulsar: routed MoE: the plan's %s slice of %.*s has no slot arm -- refusing\n",
                    op == PULSAR_TP_OP_EXPERTS ? "expert-range" : "dense", (int)G->name.len, G->name.ptr);
            return false;
        }
        map = key;
        size = UINT64_MAX / 2u;
        gate_off = pulsar_tp_expert_half_offset(c->m, G);
        up_off = pulsar_tp_expert_half_offset(c->m, U);
        down_off = pulsar_tp_expert_half_offset(c->m, D);
        mid = hi - lo;
        split = 1;
    }
    uint64_t gate_expert = 0, gate_row = 0, down_expert = 0, down_row = 0;
    if (!routed_expert_side_layout(G->type, in, mid, &gate_expert, &gate_row) ||
        !routed_expert_side_layout(D->type, mid, out, &down_expert, &down_row)) {
        fprintf(stderr, "pulsar: routed MoE: no byte layout for %.*s / %.*s at intermediate %llu -- refusing\n",
                (int)G->name.len, G->name.ptr, (int)D->name.len, D->name.ptr, (unsigned long long)mid);
        return false;
    }
    return pulsar_gpu_routed_moe_batch_tensor(c->out, c->up_out, c->mid_out, c->experts_out, map, size, gate_off,
                                              up_off, down_off, G->type, D->type, gate_expert, gate_row, down_expert,
                                              down_row, (uint32_t)in, (uint32_t)mid, (uint32_t)out, c->selected,
                                              c->weights, c->n_expert_present, c->n_expert_used, c->clamp, c->x,
                                              c->layer, c->n_tokens, split) != 0;
}

bool pulsar_moe_routed_rows(const pulsar_moe_rows_call *c) {
    const pulsar_tensor *G = c->gate, *U = c->up, *D = c->down;
    const pulsar_moe_arm arm = pulsar_moe_arm_for(G, U, D, PULSAR_ACT_BF16);
    if (arm != PULSAR_MOE_ARM_ROWS_FUSED && arm != PULSAR_MOE_ARM_ROWS_PAIR) {
        say_no_arm(G, D, "raw bf16 rows");
        return false;
    }
    const uint64_t n_expert = D->dim[2];
    /* the experts this rank reads: the plan's range of whole experts (expert parallel), else all of them */
    pulsar_tp_op op = PULSAR_TP_OP_NONE;
    uint64_t lo = 0, hi = n_expert;
    if (pulsar_tp_slice_of(c->m, G, &op, &lo, &hi, NULL) && op != PULSAR_TP_OP_EXPERTS) {
        fprintf(stderr, "pulsar: routed MoE: the plan's slice of %.*s is not a range of whole experts -- refusing\n",
                (int)G->name.len, G->name.ptr);
        return false;
    }
    pulsar_rows_moe w{};
    w.ex_lo = (int)lo;
    w.n_local = (int)(hi - lo);
    w.prompt = c->prompt;
    w.k2_gate_up = exl3_type_k2(G->type);
    w.k2_down = exl3_type_k2(D->type);
    /* one stack's table over the rank's experts: [trellis, scales] pointer pairs (exl3_expert_layout) */
    auto table = [&](const pulsar_tensor *t, const char *what) -> const void *const * {
        uint64_t stride = 0, split = 0;
        if (t->ndim != 3 || t->dim[2] != n_expert || !routed_expert_side_layout(t->type, t->dim[0], t->dim[1], &stride, &split) ||
            t->bytes != stride * n_expert) {
            fprintf(stderr, "pulsar: routed MoE: %.*s is not %llu EXL3 expert slices -- refusing\n", (int)t->name.len,
                    t->name.ptr, (unsigned long long)n_expert);
            return NULL;
        }
        /* routed_expert_side_layout's (expert bytes, row bytes) are EXL3's (stride, trellis / scales split) */
        const void *p = pulsar_gpu_weight_range_ptr(tensor_map_base(c->m, t), t->abs_offset + lo * stride,
                                                    (hi - lo) * stride, what);
        if (!p) {
            fprintf(stderr, "pulsar: routed MoE: no device copy of experts [%llu, %llu) of %.*s -- refusing\n",
                    (unsigned long long)lo, (unsigned long long)hi, (int)t->name.len, t->name.ptr);
            return NULL;
        }
        return pulsar_exl3_expert_table(p, (uint32_t)(hi - lo), stride, split);
    };
    if (arm == PULSAR_MOE_ARM_ROWS_PAIR) {
        w.gate_table = table(G, "routed experts gate");
        w.up_table = table(U, "routed experts up");
        if (!w.gate_table || !w.up_table) return false;
    } else {
        w.gate_up_table = table(G, "routed experts gate_up");
        if (!w.gate_up_table) return false;
    }
    w.down_table = table(D, "routed experts down");
    if (!w.down_table) return false;
    return pulsar_rows_moe_routed_launch(&w, c->selected, c->weights, c->x_bf16, c->n_rows, c->out, c->ws, c->ws_bytes,
                                         c->nf_flag, c->nf_code, (cudaStream_t)c->stream) == 0;
}

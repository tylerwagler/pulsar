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
 *   raw bf16 ROWS (Qwen, Nemotron-H; pulsar_moe_routed_rows):
 *       EXL3 fused gate_up + down, or an EXL3 gate + up pair + down, or BF16 stacks (L295 F7; up-only relu^2 among
 *       them, pulsar_ffn_act) -> pulsar_rows_moe_routed_launch; a LATENT block (L296 N4) wraps the routed part in its
 *       two projections (latent_in -> the routed experts on the latent -> latent_out), each through the dense front door
 *
 * What a tensor-parallel rank reads is the rank's plan's (tp_slice.cpp), not the family's: a stack the plan
 * halved (expert tensor-parallel, MXFP4 or EXL3) reads the halves registered under the engine key; a stack the plan
 * ranged (expert parallel) reads that range of whole experts.  Every arm is today's launcher, unchanged. */
#include "pulsar_engine_internal.h"
#include "exl3_trellis.h"
#include "cuda/pulsar_cuda_qwen.h"
#include "cuda/pulsar_cuda_rowops.h"
#include "cuda/mmq/pulsar_tessera.h"

pulsar_moe_arm pulsar_moe_arm_for(const pulsar_tensor *gate, const pulsar_tensor *up, const pulsar_tensor *down,
                                  pulsar_act_format act) {
    if (!gate || !down) return PULSAR_MOE_ARM_NONE;
    if (up && up->type != gate->type) return PULSAR_MOE_ARM_NONE;   /* a gate / up pair is one format */
    if (act == PULSAR_ACT_SLOT_E4M3) {
        /* at E4M3 an EXL3 side runs its own arm on both projections: it never pairs with a CUTLASS / MMQ side */
        const bool ok = up && pulsar_format_serves(gate->type, PULSAR_ROLE_EXPERT_GATE_UP, act) &&
                        pulsar_format_serves(down->type, PULSAR_ROLE_EXPERT_DOWN, act) &&
                        (exl3_type_k2(gate->type) != 0) == (exl3_type_k2(down->type) != 0);
        return ok ? PULSAR_MOE_ARM_SLOT : PULSAR_MOE_ARM_NONE;
    }
    if (act == PULSAR_ACT_ROWS_BF16 && pulsar_format_serves(down->type, PULSAR_ROLE_EXPERT_DOWN, act)) {
        /* L295 F7: BF16 experts are BF16 on all three stacks (the BF16 routed arm), never paired with EXL3 */
        if ((gate->type == PULSAR_TENSOR_BF16) != (down->type == PULSAR_TENSOR_BF16)) return PULSAR_MOE_ARM_NONE;
        if (up && gate->type == PULSAR_TENSOR_BF16) return PULSAR_MOE_ARM_ROWS_BF16;
        if (!up && pulsar_format_serves(gate->type, PULSAR_ROLE_EXPERT_GATE_UP_FUSED, act)) return PULSAR_MOE_ARM_ROWS_FUSED;
        if (up && pulsar_format_serves(gate->type, PULSAR_ROLE_EXPERT_GATE_UP, act)) return PULSAR_MOE_ARM_ROWS_PAIR;
    }
    return PULSAR_MOE_ARM_NONE;
}

pulsar_moe_arm pulsar_moe_arm_up_only(const pulsar_tensor *up, const pulsar_tensor *down, pulsar_act_format act) {
    if (!up || !down || act != PULSAR_ACT_ROWS_BF16) return PULSAR_MOE_ARM_NONE;
    const bool bf16 = up->type == PULSAR_TENSOR_BF16 && down->type == PULSAR_TENSOR_BF16 &&
                      pulsar_format_serves(up->type, PULSAR_ROLE_EXPERT_GATE_UP, act) &&
                      pulsar_format_serves(down->type, PULSAR_ROLE_EXPERT_DOWN, act);
    return bf16 ? PULSAR_MOE_ARM_ROWS_BF16 : PULSAR_MOE_ARM_NONE;
}

static void say_no_arm(const pulsar_tensor *gate, const pulsar_tensor *down, const char *against) {
    fprintf(stderr, "pulsar: routed MoE: %.*s (%s) with %.*s (%s) has no arm against %s -- refusing\n",
            (int)gate->name.len, gate->name.ptr, tensor_type_name(gate->type), (int)down->name.len, down->name.ptr,
            tensor_type_name(down->type), against);
}

bool pulsar_moe_routed_slot(const pulsar_moe_slot_call *c) {
    const pulsar_tensor *G = c->gate, *U = c->up, *D = c->down;
    if (pulsar_moe_arm_for(G, U, D, PULSAR_ACT_SLOT_E4M3) != PULSAR_MOE_ARM_SLOT) {
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
     * from the compact half stacks built at open, resolved under the engine key -- a partial the FFN exchange sums.
     * One door for both formats' halves: a half is its format's own layout at the half width, so
     * routed_expert_side_layout below is its byte model and the arm is the whole stack's */
    pulsar_tp_op op = PULSAR_TP_OP_NONE;
    uint64_t lo = 0, hi = 0;
    const void *key = NULL;
    if (pulsar_tp_slice_of(c->m, G, &op, &lo, &hi, &key)) {
        if (op != PULSAR_TP_OP_MXFP4_HALF && op != PULSAR_TP_OP_EXL3_HALF) {
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

/* L296 N4: the latent wrapper -- x_lat = bf16(latent_in x); the routed experts on x_lat into lat_f (f32, the block's
 * routed sum on the latent); out = latent_out(bf16(lat_f)).  The two bf16 roundings are the producers' (rule 3): the
 * routed experts read bf16 rows, the back projection reads bf16 rows. */
static bool latent_rows(const pulsar_moe_rows_call *c) {
    const pulsar_tensor *Li = c->latent_in, *Lo = c->latent_out;
    if (!Li || !Lo || Li->ndim != 2 || Lo->ndim != 2 || Lo->dim[0] != Li->dim[1] || Lo->dim[1] != Li->dim[0] ||
        !c->lat_f || !c->lat_x || !c->lat_y || c->shared || c->n_shared) {
        fprintf(stderr, "pulsar: routed MoE: a latent block needs both projections (hidden -> latent -> hidden), its "
                        "latent buffers and no riding shared expert -- refusing\n");
        return false;
    }
    const int H = (int)Li->dim[0], L = (int)Li->dim[1], n = c->n_rows;
    pulsar_rows_linear fc1{}, fc2{};
    if (!pulsar_linear_rows_ref(c->m, Li, H, L, c->prompt, "MoE latent in", &fc1) ||
        !pulsar_linear_rows_ref(c->m, Lo, L, H, c->prompt, "MoE latent out", &fc2))
        return false;
    if (pulsar_rows_linear_launch(&fc1, c->x_bf16, n, c->lat_f, c->lat_ws, c->lat_ws_bytes, 0) != 0 ||
        pulsar_rows_emit_bf16_launch(c->lat_f, (uint64_t)L, (uint32_t)n, (uint32_t)L, NULL, PULSAR_ROWS_SCALE_NONE,
                                     c->lat_x, (uint64_t)L, 0) != 0) {
        fprintf(stderr, "pulsar: routed MoE: the latent projection %d -> %d refused\n", H, L);
        return false;
    }
    pulsar_moe_rows_call r = *c;
    r.latent_in = r.latent_out = NULL;
    r.x_bf16 = c->lat_x;
    r.out = c->lat_f;   /* the routed sum on the latent (the fc1 output is consumed: lat_x holds its bf16 rows) */
    if (!pulsar_moe_routed_rows(&r)) return false;
    if (pulsar_rows_emit_bf16_launch(c->lat_f, (uint64_t)L, (uint32_t)n, (uint32_t)L, NULL, PULSAR_ROWS_SCALE_NONE,
                                     c->lat_y, (uint64_t)L, 0) != 0 ||
        pulsar_rows_linear_launch(&fc2, c->lat_y, n, c->out, c->lat_ws, c->lat_ws_bytes, 0) != 0) {
        fprintf(stderr, "pulsar: routed MoE: the latent back projection %d -> %d refused\n", L, H);
        return false;
    }
    return true;
}

bool pulsar_moe_routed_rows(const pulsar_moe_rows_call *c) {
    if (c->latent_in || c->latent_out) return latent_rows(c);
    if (c->tessera_gate) {
        /* L255: Tessera planes carry no format the registry names -- the family admitted them plane by plane
         * (admit_tessera) and built the three descriptors; the arm takes every expert (no range: the family refuses
         * the call under tensor parallelism), so the plan is not consulted */
        pulsar_rows_moe w{};
        w.tessera_gate = c->tessera_gate;
        w.tessera_up = c->tessera_up;
        w.tessera_down = c->tessera_down;
        w.ex_lo = 0;
        w.n_local = c->tessera_gate->E;
        w.n_expert = c->tessera_gate->E;
        w.top_k = (int)c->top_k;
        w.in_dim = c->tessera_gate->K;
        w.mid_dim = c->tessera_gate->N;
        w.out_dim = c->tessera_down->N;
        w.prompt = c->prompt;
        w.shared = c->shared;   /* the launcher refuses them: only the EXL3 front door carries shared experts */
        w.n_shared = c->n_shared;
        return pulsar_rows_moe_routed_launch(&w, c->selected, c->weights, c->x_bf16, c->n_rows, c->out, c->ws,
                                             c->ws_bytes, c->nf_flag, c->nf_code, 0) == 0;
    }
    const pulsar_tensor *G = c->gate, *U = c->up, *D = c->down;
    /* L295 F7: a non-gated activation (RELU2) reads up-only experts, which only the BF16 arm runs */
    const pulsar_ffn_act act = (pulsar_ffn_act)c->act;
    if (act != PULSAR_FFN_ACT_SWIGLU) {
        if (act != PULSAR_FFN_ACT_RELU2 || G ||
            pulsar_moe_arm_up_only(U, D, PULSAR_ACT_ROWS_BF16) != PULSAR_MOE_ARM_ROWS_BF16) {
            fprintf(stderr, "pulsar: routed MoE: activation %d runs BF16 up-only experts (no gate stack) alone -- "
                            "refusing\n", (int)act);
            return false;
        }
    }
    const pulsar_tensor *G0 = G ? G : U;   /* the first projection: its input width, its plan slice */
    const pulsar_moe_arm arm = act == PULSAR_FFN_ACT_RELU2 ? PULSAR_MOE_ARM_ROWS_BF16
                                                           : pulsar_moe_arm_for(G, U, D, PULSAR_ACT_ROWS_BF16);
    if (arm != PULSAR_MOE_ARM_ROWS_FUSED && arm != PULSAR_MOE_ARM_ROWS_PAIR && arm != PULSAR_MOE_ARM_ROWS_BF16) {
        say_no_arm(G0, D, "raw bf16 rows");
        return false;
    }
    const uint64_t n_expert = D->dim[2];
    /* the experts this rank reads: the plan's range of whole experts (expert parallel), else all of them */
    pulsar_tp_op op = PULSAR_TP_OP_NONE;
    uint64_t lo = 0, hi = n_expert;
    if (pulsar_tp_slice_of(c->m, G0, &op, &lo, &hi, NULL) && op != PULSAR_TP_OP_EXPERTS) {
        fprintf(stderr, "pulsar: routed MoE: the plan's slice of %.*s is not a range of whole experts -- refusing\n",
                (int)G0->name.len, G0->name.ptr);
        return false;
    }
    pulsar_rows_moe w{};
    w.ex_lo = (int)lo;
    w.n_local = (int)(hi - lo);
    w.n_expert = (int)n_expert;
    w.top_k = (int)c->top_k;
    w.in_dim = (int)G0->dim[0];
    w.mid_dim = (int)D->dim[0];
    w.out_dim = (int)D->dim[1];
    w.prompt = c->prompt;
    w.shared = c->shared;
    w.n_shared = c->n_shared;
    w.k2_gate_up = exl3_type_k2(G0->type);
    w.act = act;
    w.k2_down = exl3_type_k2(D->type);
    /* one stack's experts [lo, hi) on the device: their base pointer, and their stride / split point
     * (routed_expert_side_layout: EXL3's [trellis | scales], BF16's [out][in] rows) */
    auto stack = [&](const pulsar_tensor *t, const char *what, uint64_t *stride, uint64_t *split) -> const void * {
        if (t->ndim != 3 || t->dim[2] != n_expert || !routed_expert_side_layout(t->type, t->dim[0], t->dim[1], stride, split) ||
            t->bytes != *stride * n_expert) {
            fprintf(stderr, "pulsar: routed MoE: %.*s is not %llu %s expert slices -- refusing\n", (int)t->name.len,
                    t->name.ptr, (unsigned long long)n_expert, tensor_type_name(t->type));
            return NULL;
        }
        const void *p = pulsar_gpu_weight_range_ptr(tensor_map_base(c->m, t), t->abs_offset + lo * *stride,
                                                    (hi - lo) * *stride, what);
        if (!p)
            fprintf(stderr, "pulsar: routed MoE: no device copy of experts [%llu, %llu) of %.*s -- refusing\n",
                    (unsigned long long)lo, (unsigned long long)hi, (int)t->name.len, t->name.ptr);
        return p;
    };
    if (arm == PULSAR_MOE_ARM_ROWS_BF16) {
        /* L295 F7: the BF16 stacks themselves (pulsar_cuda_bf16_rows.cuh reads [expert][out][in] by stride) */
        uint64_t s = 0, r = 0;
        w.bf16_gate = G ? stack(G, "routed experts gate", &s, &r) : NULL;   /* NULL: up-only (RELU2) */
        w.bf16_up = stack(U, "routed experts up", &s, &r);
        w.bf16_down = stack(D, "routed experts down", &s, &r);
        if ((G && !w.bf16_gate) || !w.bf16_up || !w.bf16_down) return false;
        return pulsar_rows_moe_routed_launch(&w, c->selected, c->weights, c->x_bf16, c->n_rows, c->out, c->ws,
                                             c->ws_bytes, c->nf_flag, c->nf_code, 0) == 0;
    }
    /* one stack's table over the rank's experts: [trellis, scales] pointer pairs (exl3_expert_layout) */
    auto table = [&](const pulsar_tensor *t, const char *what) -> const void *const * {
        uint64_t stride = 0, split = 0;
        const void *p = stack(t, what, &stride, &split);
        return p ? pulsar_exl3_expert_table(p, (uint32_t)(hi - lo), stride, split) : NULL;
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
                                         c->nf_flag, c->nf_code, 0) == 0;   /* the per-thread default stream */
}

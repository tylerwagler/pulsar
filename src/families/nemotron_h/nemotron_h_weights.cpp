/* nemotron_h_weights.cpp -- the Nemotron-H family's weights: every tensor the shape declares, bound by its bind record
 * through the core's role binder (src/engine/role_bind.h), its format admitted through the core's front doors
 * (weight_format.cpp), its bytes priced (nemotron_h.h section 2).
 *
 * The formats.  The forward hands its linears raw bf16 rows (pulsar_act_kind ROWS), so every linear -- the Mamba-2
 * in / out projections, attention q / k / v / o, the latent fc1 / fc2, the drafter's eh_proj -- takes what the
 * registry serves a dense linear at bf16 rows (BF16, an EXL3 dense-arm rate, mxfp8_lt), the shared expert what it
 * serves a shared expert, the head what it serves a dense linear or BF16 as stored (E5's head op reads it).  The routed
 * stacks are admitted per stack by the registry's routed roles at bf16 rows (up as the gate/up side, down as down:
 * BF16 or EXL3), and their COMBINATION -- an up-only (ungated) relu^2 expert -- by the core's up-only arm
 * (moe.cpp pulsar_moe_arm_up_only: BF16 up + down, N4; EXL3 up + down, the routed front door's up-only form), so a
 * container whose experts no arm runs is refused here, once, not at its first step.  Norms, the conv, dt_bias / A_log
 * / D and the router are read as stored: bf16, the router's selection bias f32. */
/** @file nemotron_h_weights.cpp
 *  @brief Nemotron-H's weights: every tensor bound by role, its format admitted and priced; the unselected drafter
 *  unloaded. */
#include "nemotron_h.h"

namespace {

/// The binder's and the format checks' owner name.
const char *const OWNER = PULSAR_NEMOTRON_H_ARCH;
/// The trunk layers' block name in role keys.
const char *const LAYERS = "layers";
/// The MTP module's block name in role keys.
const char *const MTP = "mtp";

/// Read as stored: bf16 only (norms, the conv, the SSM parameters, the router).
const pulsar_role_admit A_BF16 = {PULSAR_ROLE_PLAIN(PULSAR_TENSOR_BF16), false, PULSAR_ROLE_DENSE, 0};
/// Read as stored: f32 only (the router's selection bias).
const pulsar_role_admit A_F32 = {PULSAR_ROLE_PLAIN(PULSAR_TENSOR_F32), false, PULSAR_ROLE_DENSE, 0};
/// A dense linear: whatever the registry serves at bf16 rows.
const pulsar_role_admit A_LINEAR = {0, true, PULSAR_ROLE_DENSE, PULSAR_ACTS(PULSAR_ACT_ROWS_BF16)};
/// The head: bf16 as stored, or a dense-arm format at bf16 rows.
const pulsar_role_admit A_HEAD = {PULSAR_ROLE_PLAIN(PULSAR_TENSOR_BF16), true, PULSAR_ROLE_DENSE,
                                  PULSAR_ACTS(PULSAR_ACT_ROWS_BF16)};
/// The shared expert's linears at bf16 rows.
const pulsar_role_admit A_SHARED = {0, true, PULSAR_ROLE_SHARED_EXPERT, PULSAR_ACTS(PULSAR_ACT_ROWS_BF16)};
/// The routed up stack at bf16 rows (admitted as a gate / up stack; the experts have no gate).
const pulsar_role_admit A_EXP_UP = {0, true, PULSAR_ROLE_EXPERT_GATE_UP, PULSAR_ACTS(PULSAR_ACT_ROWS_BF16)};
/// The routed down stack at bf16 rows.
const pulsar_role_admit A_EXP_DOWN = {0, true, PULSAR_ROLE_EXPERT_DOWN, PULSAR_ACTS(PULSAR_ACT_ROWS_BF16)};

/* One pulsar layer: a trunk layer (`layers`, il) or the drafter's (`mtp`, 0).  mamba / moe from the plan. */
/** @param b      the open binder
 *  @param s      the shape (widths)
 *  @param block  the role block name (LAYERS or MTP)
 *  @param il     the layer index within that block
 *  @param mamba  bind the Mamba-2 mixer; false = the attention mixer
 *  @param moe    also bind the latent MoE
 *  @param L      receives the layer's tensors */
void bind_block(pulsar_role_binder *b, const pulsar_nemotron_h_shape *s, const char *block, int32_t il, bool mamba,
                bool moe, pulsar_nemotron_h_block *L) {
    const uint64_t E = s->n_embd;
    auto k = [&](const char *role, int32_t index = -1) { return pulsar_role_in(role, block, il, index); };
    if (mamba) {
        const uint64_t DI = (uint64_t)s->ssm_n_head * s->ssm_head_dim;
        const uint64_t CD = DI + 2ull * s->ssm_groups * s->ssm_state;   /* the conv's channels: x | B | C */
        L->ssm_pre_norm = pulsar_role_bind(b, k("ssm_pre_norm"), A_BF16, 1, E);
        L->ssm_in = pulsar_role_bind(b, k("ssm_in"), A_LINEAR, 2, E, DI + CD + s->ssm_n_head);
        L->ssm_conv = pulsar_role_bind(b, k("ssm_conv"), A_BF16, 2, s->conv_k, CD);
        L->ssm_conv_bias = pulsar_role_bind(b, k("ssm_conv_bias"), A_BF16, 1, CD);
        L->ssm_dt_bias = pulsar_role_bind(b, k("ssm_dt_bias"), A_BF16, 1, s->ssm_n_head);
        L->ssm_a_log = pulsar_role_bind(b, k("ssm_a_log"), A_BF16, 1, s->ssm_n_head);
        L->ssm_d = pulsar_role_bind(b, k("ssm_d"), A_BF16, 1, s->ssm_n_head);
        L->ssm_out_norm = pulsar_role_bind(b, k("ssm_out_norm"), A_BF16, 1, DI);
        L->ssm_out = pulsar_role_bind(b, k("ssm_out"), A_LINEAR, 2, DI, E);
    } else {
        const uint64_t Q = (uint64_t)s->n_head * s->head_dim, KV = (uint64_t)s->n_head_kv * s->head_dim;
        L->attn_norm = pulsar_role_bind(b, k("attn_norm"), A_BF16, 1, E);
        L->attn_q = pulsar_role_bind(b, k("attn_q"), A_LINEAR, 2, E, Q);
        L->attn_k = pulsar_role_bind(b, k("attn_k"), A_LINEAR, 2, E, KV);
        L->attn_v = pulsar_role_bind(b, k("attn_v"), A_LINEAR, 2, E, KV);
        L->attn_o = pulsar_role_bind(b, k("attn_o"), A_LINEAR, 2, Q, E);
    }
    if (!moe) return;
    const uint64_t X = s->n_expert, F = s->n_ff_exp, Lt = s->latent;
    L->ffn_norm = pulsar_role_bind(b, k("ffn_norm"), A_BF16, 1, E);
    L->router = pulsar_role_bind(b, k("router"), A_BF16, 2, E, X);
    L->router_select_bias = pulsar_role_bind(b, k("router_select_bias"), A_F32, 1, X);
    L->latent_in = pulsar_role_bind(b, k("latent_in"), A_LINEAR, 2, E, Lt);
    L->latent_out = pulsar_role_bind(b, k("latent_out"), A_LINEAR, 2, Lt, E);
    L->exp_up = pulsar_role_bind(b, k("expert_up"), A_EXP_UP, 3, Lt, F, X);
    L->exp_down = pulsar_role_bind(b, k("expert_down"), A_EXP_DOWN, 3, F, Lt, X);
    if (L->exp_up && L->exp_down &&
        pulsar_moe_arm_up_only(L->exp_up, L->exp_down, PULSAR_ACT_ROWS_BF16) == PULSAR_MOE_ARM_NONE) {
        fprintf(stderr, "pulsar: %s: %s %d: up-only relu^2 experts in %s / %s have no arm (BF16 or EXL3, up and down "
                        "in one format) -- refusing\n", OWNER, block, il,
                tensor_type_name(L->exp_up->type), tensor_type_name(L->exp_down->type));
        pulsar_load_refuse();
    }
    L->sh_up = pulsar_role_bind(b, k("shared_up", 0), A_SHARED, 2, E, s->n_ff_shared);
    L->sh_down = pulsar_role_bind(b, k("shared_down", 0), A_SHARED, 2, s->n_ff_shared, E);
}

}  // namespace

/** @param m  the container
 *  @param w  the family state: its shape, drafting and block are read; the tensors and the census are written
 *  @return true = bound; false = refused (said) */
bool pulsar_nemotron_h_bind(const pulsar_model *m, pulsar_nemotron_h_model *w) {
    const pulsar_nemotron_h_shape *s = &w->shape;
    pulsar_role_binder *b = pulsar_role_binder_open(m, OWNER);
    const uint64_t E = s->n_embd, V = s->n_vocab;
    w->embed = pulsar_role_bind(b, pulsar_role_top("embed"), A_BF16, 2, E, V);
    w->output_norm = pulsar_role_bind(b, pulsar_role_top("output_norm"), A_BF16, 1, E);
    w->head = pulsar_role_bind(b, pulsar_role_top("head"), A_HEAD, 2, E, V);
    for (uint32_t il = 0; il < s->n_layer; il++)
        bind_block(b, s, LAYERS, (int32_t)il, s->mamba_layers >> il & 1u, s->moe_layers >> il & 1u, &w->layer[il]);
    /* the drafter: one {attention, MoE} layer plus its input side and its out norm, when the open selected it (L298:
     * otherwise declared unloaded -- never staged, never priced) */
    if (s->mtp_written && !w->drafting) {
        const uint64_t n = pulsar_role_binder_unload(b, MTP);
        fprintf(stderr, "pulsar: %s: the MTP module (%llu tensors) is not the selected drafter: unloaded\n", OWNER,
                (unsigned long long)n);
    } else if (s->mtp_written) {
        pulsar_nemotron_h_draft *D = &w->mtp;
        bind_block(b, s, MTP, 0, false, true, &D->block);
        D->embed_norm = pulsar_role_bind(b, pulsar_role_in("draft_embed_norm", MTP, 0), A_BF16, 1, E);
        D->hidden_norm = pulsar_role_bind(b, pulsar_role_in("draft_hidden_norm", MTP, 0), A_BF16, 1, E);
        D->in_proj = pulsar_role_bind(b, pulsar_role_in("draft_in_proj", MTP, 0), A_LINEAR, 2, 2 * E, E);
        D->out_norm = pulsar_role_bind(b, pulsar_role_in("draft_out_norm", MTP, 0), A_BF16, 1, E);
    }
    /* L298: the LiLiCorr block drafter (block `drafter`) and its candidate head (block `lilicorr`) when the open
     * selected it, else unloaded */
    if (s->block_written) {
        if (w->block) {
            (void)pulsar_block_drafter_bind(b, w->block);
        } else {
            const uint64_t n = pulsar_role_binder_unload(b, "drafter") + pulsar_role_binder_unload(b, "lilicorr");
            fprintf(stderr, "pulsar: %s: the LiLiCorr drafter (%llu tensors) is not the selected drafter: unloaded\n",
                    OWNER, (unsigned long long)n);
        }
    }
    /* the census: every directory entry declared, every byte priced (the extras refused by name) */
    return pulsar_role_binder_close(b, &w->census);
}

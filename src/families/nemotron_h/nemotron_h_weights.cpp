/* nemotron_h_weights.cpp -- the Nemotron-H family's weights: every tensor the shape declares, bound by its bind record
 * through the core's role binder (src/engine/role_bind.h), its format admitted through the core's front doors
 * (weight_format.cpp), its bytes priced (nemotron_h.h section 2).
 *
 * The formats.  The forward hands its linears raw bf16 rows (pulsar_act_kind ROWS), so every linear -- the Mamba-2
 * in / out projections, attention q / k / v / o, the latent fc1 / fc2, the drafter's eh_proj -- takes what the
 * registry serves a dense linear at bf16 rows (BF16, an EXL3 dense-arm rate, mxfp8_lt), the shared expert what it
 * serves a shared expert, the head what it serves a dense linear or BF16 as stored (E5's head op reads it).  The routed
 * stacks are admitted per stack by the registry's routed roles at bf16 rows (up as the gate/up side, down as down:
 * BF16 today).  Their COMBINATION -- an up-only (ungated) relu^2 expert -- has no arm in any format yet (N4 builds the
 * BF16 one; a quant format's comes with the L296 format verdict), so no pair check runs here: the forward, not the
 * load, owns that refusal until N4.  Norms, the conv, dt_bias / A_log / D and the router are read as stored: bf16, the
 * router's selection bias f32. */
#include "nemotron_h.h"

namespace {

const char *const OWNER = PULSAR_NEMOTRON_H_ARCH;
const char *const LAYERS = "layers";
const char *const MTP = "mtp";

const pulsar_role_admit A_BF16 = {PULSAR_ROLE_PLAIN(PULSAR_TENSOR_BF16), false, PULSAR_ROLE_DENSE, 0};
const pulsar_role_admit A_F32 = {PULSAR_ROLE_PLAIN(PULSAR_TENSOR_F32), false, PULSAR_ROLE_DENSE, 0};
const pulsar_role_admit A_LINEAR = {0, true, PULSAR_ROLE_DENSE, PULSAR_ACTS(PULSAR_ACT_ROWS_BF16)};
const pulsar_role_admit A_HEAD = {PULSAR_ROLE_PLAIN(PULSAR_TENSOR_BF16), true, PULSAR_ROLE_DENSE,
                                  PULSAR_ACTS(PULSAR_ACT_ROWS_BF16)};
const pulsar_role_admit A_SHARED = {0, true, PULSAR_ROLE_SHARED_EXPERT, PULSAR_ACTS(PULSAR_ACT_ROWS_BF16)};
const pulsar_role_admit A_EXP_UP = {0, true, PULSAR_ROLE_EXPERT_GATE_UP, PULSAR_ACTS(PULSAR_ACT_ROWS_BF16)};
const pulsar_role_admit A_EXP_DOWN = {0, true, PULSAR_ROLE_EXPERT_DOWN, PULSAR_ACTS(PULSAR_ACT_ROWS_BF16)};

/* One pulsar layer: a trunk layer (`layers`, il) or the drafter's (`mtp`, 0).  mamba / moe from the plan. */
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
    L->sh_up = pulsar_role_bind(b, k("shared_up", 0), A_SHARED, 2, E, s->n_ff_shared);
    L->sh_down = pulsar_role_bind(b, k("shared_down", 0), A_SHARED, 2, s->n_ff_shared, E);
}

}  // namespace

bool pulsar_nemotron_h_bind(const pulsar_model *m, pulsar_nemotron_h_model *w) {
    const pulsar_nemotron_h_shape *s = &w->shape;
    pulsar_role_binder *b = pulsar_role_binder_open(m, OWNER);
    const uint64_t E = s->n_embd, V = s->n_vocab;
    w->embed = pulsar_role_bind(b, pulsar_role_top("embed"), A_BF16, 2, E, V);
    w->output_norm = pulsar_role_bind(b, pulsar_role_top("output_norm"), A_BF16, 1, E);
    w->head = pulsar_role_bind(b, pulsar_role_top("head"), A_HEAD, 2, E, V);
    for (uint32_t il = 0; il < s->n_layer; il++)
        bind_block(b, s, LAYERS, (int32_t)il, s->mamba_layers >> il & 1u, s->moe_layers >> il & 1u, &w->layer[il]);
    /* the drafter: one {attention, MoE} layer plus its input side and its out norm, when the container holds it */
    if (s->mtp_written) {
        pulsar_nemotron_h_draft *D = &w->mtp;
        bind_block(b, s, MTP, 0, false, true, &D->block);
        D->embed_norm = pulsar_role_bind(b, pulsar_role_in("draft_embed_norm", MTP, 0), A_BF16, 1, E);
        D->hidden_norm = pulsar_role_bind(b, pulsar_role_in("draft_hidden_norm", MTP, 0), A_BF16, 1, E);
        D->in_proj = pulsar_role_bind(b, pulsar_role_in("draft_in_proj", MTP, 0), A_LINEAR, 2, 2 * E, E);
        D->out_norm = pulsar_role_bind(b, pulsar_role_in("draft_out_norm", MTP, 0), A_BF16, 1, E);
    }
    /* the census: every directory entry declared, every byte priced (the extras refused by name) */
    return pulsar_role_binder_close(b, &w->census);
}

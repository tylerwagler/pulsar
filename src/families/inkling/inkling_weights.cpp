/* inkling_weights.cpp -- the Inkling family's weights: every tensor the shape declares, bound by its bind record
 * through the core's role binder (src/engine/role_bind.h), its format admitted through the core's front doors
 * (weight_format.cpp), its bytes priced (inkling.h section 2).
 *
 * The formats.  The forward hands its linears raw bf16 rows (pulsar_act_kind ROWS, as Qwen's), so every linear --
 * attention q / k / v / r / o, the dense MLP, the shared experts, the drafter's input projection -- takes what the
 * registry serves a dense linear (or a shared expert) at bf16 rows: EXL3 at a dense-arm rate, or mxfp8_lt.  The
 * routed experts take the registry's split gate / up pair and down arms at bf16 rows (EXL3 at a pair / down rate);
 * a K5 gate / up and a bf16 linear at bf16 rows have no arm today and are refused by name (L295 F7).  The head is
 * bf16 as the recipe writes it (the head op, E5, reads it) or a dense-arm format.  Norms, short convs, the relative
 * bases, the router and the tower are read as stored: bf16, the router's selection bias and output scale f32. */
#include "inkling.h"

namespace {

const char *const OWNER = PULSAR_INKLING_ARCH;
const char *const LAYERS = "layers";
const char *const MTP = "mtp";

const pulsar_role_admit A_BF16 = {PULSAR_ROLE_PLAIN(PULSAR_TENSOR_BF16), false, PULSAR_ROLE_DENSE, 0};
const pulsar_role_admit A_F32 = {PULSAR_ROLE_PLAIN(PULSAR_TENSOR_F32), false, PULSAR_ROLE_DENSE, 0};
const pulsar_role_admit A_LINEAR = {0, true, PULSAR_ROLE_DENSE, PULSAR_ACTS(PULSAR_ACT_ROWS_BF16)};
const pulsar_role_admit A_HEAD = {PULSAR_ROLE_PLAIN(PULSAR_TENSOR_BF16), true, PULSAR_ROLE_DENSE,
                                  PULSAR_ACTS(PULSAR_ACT_ROWS_BF16)};
const pulsar_role_admit A_SHARED = {0, true, PULSAR_ROLE_SHARED_EXPERT, PULSAR_ACTS(PULSAR_ACT_ROWS_BF16)};
const pulsar_role_admit A_EXP_GATE_UP = {0, true, PULSAR_ROLE_EXPERT_GATE_UP, PULSAR_ACTS(PULSAR_ACT_ROWS_BF16)};
const pulsar_role_admit A_EXP_DOWN = {0, true, PULSAR_ROLE_EXPERT_DOWN, PULSAR_ACTS(PULSAR_ACT_ROWS_BF16)};

/* One transformer block: a trunk layer (`layers`, il) or a drafter layer (`mtp`, n).  window / moe from the plan. */
void bind_block(pulsar_role_binder *b, const pulsar_inkling_shape *s, const char *block, int32_t il, bool window,
                bool moe, pulsar_inkling_block *L) {
    const uint64_t E = s->n_embd;
    const uint64_t nq = window ? s->swa_n_head : s->n_head, nkv = window ? s->swa_n_head_kv : s->n_head_kv;
    const uint64_t hd = window ? s->swa_head_dim : s->head_dim, extent = window ? s->window : s->rel_extent;
    auto k = [&](const char *role, int32_t index = -1) { return pulsar_role_in(role, block, il, index); };
    L->attn_norm = pulsar_role_bind(b, k("attn_norm"), A_BF16, 1, E);
    L->attn_q = pulsar_role_bind(b, k("attn_q"), A_LINEAR, 2, E, nq * hd);
    L->attn_k = pulsar_role_bind(b, k("attn_k"), A_LINEAR, 2, E, nkv * hd);
    L->attn_v = pulsar_role_bind(b, k("attn_v"), A_LINEAR, 2, E, nkv * hd);
    L->attn_rel_proj = pulsar_role_bind(b, k("attn_rel_proj"), A_LINEAR, 2, E, nq * s->d_rel);
    L->attn_o = pulsar_role_bind(b, k("attn_o"), A_LINEAR, 2, nq * hd, E);
    L->attn_q_norm = pulsar_role_bind(b, k("attn_q_norm"), A_BF16, 1, hd);
    L->attn_k_norm = pulsar_role_bind(b, k("attn_k_norm"), A_BF16, 1, hd);
    L->attn_rel_basis = pulsar_role_bind(b, k("attn_rel_basis"), A_BF16, 2, extent, s->d_rel);
    L->conv_k = pulsar_role_bind(b, k("short_conv_k"), A_BF16, 2, s->conv_k, nkv * hd);
    L->conv_v = pulsar_role_bind(b, k("short_conv_v"), A_BF16, 2, s->conv_k, nkv * hd);
    L->conv_attn_out = pulsar_role_bind(b, k("short_conv_attn_out"), A_BF16, 2, s->conv_k, E);
    L->conv_ffn_out = pulsar_role_bind(b, k("short_conv_ffn_out"), A_BF16, 2, s->conv_k, E);
    L->ffn_norm = pulsar_role_bind(b, k("ffn_norm"), A_BF16, 1, E);
    if (!moe) {
        L->ffn_gate = pulsar_role_bind(b, k("ffn_gate"), A_LINEAR, 2, E, s->n_ff_dense);
        L->ffn_up = pulsar_role_bind(b, k("ffn_up"), A_LINEAR, 2, E, s->n_ff_dense);
        L->ffn_down = pulsar_role_bind(b, k("ffn_down"), A_LINEAR, 2, s->n_ff_dense, E);
        L->ffn_out_scale = pulsar_role_bind(b, k("ffn_out_scale"), A_BF16, 1, 1);
        return;
    }
    const uint64_t F = s->n_ff_exp, X = s->n_expert;
    L->router = pulsar_role_bind(b, k("router"), A_BF16, 2, E, X + s->n_shared);
    L->router_select_bias = pulsar_role_bind(b, k("router_select_bias"), A_F32, 1, X);
    L->router_out_scale = pulsar_role_bind(b, k("router_out_scale"), A_F32, 1, 1);
    L->exp_gate = pulsar_role_bind(b, k("expert_gate"), A_EXP_GATE_UP, 3, E, F, X);
    L->exp_up = pulsar_role_bind(b, k("expert_up"), A_EXP_GATE_UP, 3, E, F, X);
    L->exp_down = pulsar_role_bind(b, k("expert_down"), A_EXP_DOWN, 3, F, E, X);
    if (L->exp_gate && L->exp_up && L->exp_down &&
        !pulsar_format_moe_combo(L->exp_gate, L->exp_up, L->exp_down, PULSAR_ACT_ROWS_BF16, OWNER))
        pulsar_load_refuse();
    for (uint32_t j = 0; j < s->n_shared; j++) {
        L->sh_gate[j] = pulsar_role_bind(b, k("shared_gate", (int32_t)j), A_SHARED, 2, E, F);
        L->sh_up[j] = pulsar_role_bind(b, k("shared_up", (int32_t)j), A_SHARED, 2, E, F);
        L->sh_down[j] = pulsar_role_bind(b, k("shared_down", (int32_t)j), A_SHARED, 2, F, E);
    }
}

/* The hmlp tower: linear i maps the previous width to the next -- the first reads a flattened patch
 * (patch^2 x channels x temporal), the last writes n_embd -- with a norm between linears and a final norm after.
 * The inner widths are the checkpoint's own (the config does not name them): each is the stored linear's output,
 * and the next linear's input is checked against it. */
void bind_tower(pulsar_role_binder *b, const pulsar_inkling_shape *s, pulsar_inkling_model *w) {
    uint64_t in = (uint64_t)s->vis_patch * s->vis_patch * s->vis_channels * s->vis_temporal;
    for (uint32_t i = 0; i < s->vis_layers; i++) {
        const bool last = i + 1 == s->vis_layers;
        const pulsar_role_key k = pulsar_role_top("vision_linear", (int32_t)i);
        const pulsar_tensor *t = pulsar_role_find(b, k);
        const uint64_t out = last ? s->n_embd : t && t->ndim == 2 ? t->dim[1] : 0;
        w->vis_linear[i] = pulsar_role_bind(b, k, A_BF16, 2, in, out);
        if (!last) w->vis_norm[i] = pulsar_role_bind(b, pulsar_role_top("vision_norm", (int32_t)i), A_BF16, 1, out);
        in = out;
    }
    w->vis_final_norm = pulsar_role_bind(b, pulsar_role_top("vision_final_norm"), A_BF16, 1, s->n_embd);
}

}  // namespace

bool pulsar_inkling_bind(const pulsar_model *m, pulsar_inkling_model *w) {
    const pulsar_inkling_shape *s = &w->shape;
    pulsar_role_binder *b = pulsar_role_binder_open(m, OWNER);
    const uint64_t E = s->n_embd, V = s->n_vocab;
    w->embed = pulsar_role_bind(b, pulsar_role_top("embed"), A_BF16, 2, E, V);
    w->embed_norm = pulsar_role_bind(b, pulsar_role_top("embed_norm"), A_BF16, 1, E);
    w->output_norm = pulsar_role_bind(b, pulsar_role_top("output_norm"), A_BF16, 1, E);
    w->head = pulsar_role_bind(b, pulsar_role_top("head"), A_HEAD, 2, E, V);
    for (uint32_t il = 0; il < s->n_layer; il++)
        bind_block(b, s, LAYERS, (int32_t)il, s->window_layers >> il & 1u, il >= s->n_dense, &w->layer[il]);
    /* the drafter: n chained layers, each a dense block plus its input side -- all of them when the container says
     * it holds the drafter (the builder writes all or none) */
    for (uint32_t n = 0; s->mtp_written && n < s->n_mtp; n++) {
        pulsar_inkling_draft_layer *D = &w->mtp[n];
        bind_block(b, s, MTP, (int32_t)n, s->mtp_window_layers >> n & 1u, false, &D->block);
        D->embed_norm = pulsar_role_bind(b, pulsar_role_in("draft_embed_norm", MTP, (int32_t)n), A_BF16, 1, E);
        D->hidden_norm = pulsar_role_bind(b, pulsar_role_in("draft_hidden_norm", MTP, (int32_t)n), A_BF16, 1, E);
        D->in_proj = pulsar_role_bind(b, pulsar_role_in("draft_in_proj", MTP, (int32_t)n), A_LINEAR, 2, 2 * E, E);
    }
    if (s->vision_written) bind_tower(b, s, w);
    /* the census: every directory entry declared, every byte priced (the extras refused by name) */
    return pulsar_role_binder_close(b, &w->census);
}

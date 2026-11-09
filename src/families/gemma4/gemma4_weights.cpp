/* gemma4_weights.cpp -- the Gemma 4 family's weights: every tensor the shape declares, bound by its bind record
 * through the core's role binder (src/engine/role_bind.h), its format admitted through the core's front doors
 * (weight_format.cpp), its bytes priced (gemma4.h section 2).
 *
 * The formats.  The forward hands its linears raw bf16 rows (pulsar_act_kind ROWS, as Qwen's and Inkling's), so every
 * linear -- attention q / k / v / o and the dense MLP -- takes what the registry serves a dense linear at bf16 rows,
 * and the routed experts take the registry's split gate / up pair and down arms at bf16 rows, checked as a combo
 * (pulsar_format_moe_combo).  The dense MLP and the experts are bound at their STORED widths (gemma4.h).  The head is
 * the pack's EXL3 one, a dense linear at bf16 rows (the embedding it was quantized from is bound for the lookups).
 * Norms, the router, its two scales, the per-layer scalar and the embedding are read as stored: bf16. */
/** @file gemma4_weights.cpp
 *  @brief Gemma 4's weights: every tensor bound by role, its format admitted and priced. */
#include "gemma4.h"
#include "engine/spec_block.h"   /* L338 P3: the block drafter's bind */

namespace {

/// The binder's and the format checks' owner name.
const char *const OWNER = PULSAR_GEMMA4_ARCH;
/// The layers' block name in role keys.
const char *const LAYERS = "layers";

/// Read as stored: bf16 only (norms, router, scales, the scalar, the embedding).
const pulsar_role_admit A_BF16 = {PULSAR_ROLE_PLAIN(PULSAR_TENSOR_BF16), false, PULSAR_ROLE_DENSE, 0};
/// A dense linear: whatever the registry serves at bf16 rows.
const pulsar_role_admit A_LINEAR = {0, true, PULSAR_ROLE_DENSE, PULSAR_ACTS(PULSAR_ACT_ROWS_BF16)};
/// A routed gate or up stack at bf16 rows.
const pulsar_role_admit A_EXP_GATE_UP = {0, true, PULSAR_ROLE_EXPERT_GATE_UP, PULSAR_ACTS(PULSAR_ACT_ROWS_BF16)};
/// A routed down stack at bf16 rows.
const pulsar_role_admit A_EXP_DOWN = {0, true, PULSAR_ROLE_EXPERT_DOWN, PULSAR_ACTS(PULSAR_ACT_ROWS_BF16)};
/// L338 P2: the decision head, read as stored: f32 only (the host reads it).
const pulsar_role_admit A_F32 = {PULSAR_ROLE_PLAIN(PULSAR_TENSOR_F32), false, PULSAR_ROLE_DENSE, 0};
/// L338: the LoRA pairs: f16 only (the runtime LoRA's tensor-core operand; the builder writes them so).
const pulsar_role_admit A_F16 = {PULSAR_ROLE_PLAIN(PULSAR_TENSOR_F16), false, PULSAR_ROLE_DENSE, 0};

/** L338 P2: one projection's LoRA pair at its logical widths: A [r][in] (ne in, r), B [out][r] (ne r, out).
 *  @param b      the open binder
 *  @param target the projection's bind role (attn_q ...): the pair is `<target>_lora_a` / `_lora_b`
 *  @param il     the layer
 *  @param r      the rank
 *  @param in     the projection's logical input width
 *  @param out    its logical output width
 *  @param L      receives the pair */
void bind_lora(pulsar_role_binder *b, const char *target, int32_t il, uint64_t r, uint64_t in, uint64_t out,
               pulsar_gemma4_lora *L) {
    char ra[48], rb[48];
    snprintf(ra, sizeof(ra), "%s_lora_a", target);
    snprintf(rb, sizeof(rb), "%s_lora_b", target);
    L->a = pulsar_role_bind(b, pulsar_role_in(ra, LAYERS, il), A_F16, 2, in, r);
    L->b = pulsar_role_bind(b, pulsar_role_in(rb, LAYERS, il), A_F16, 2, r, out);
}

/** L338 P2: the adapter, when the container carries one (pulsar.adapter): its rank and scale from the kv, every
 *  targeted projection's LoRA pair at the logical widths (the dense MLP at n_ff, which the engine pads to the stored
 *  width), the decision head.  A container without one binds nothing here.
 *  @param m  the container
 *  @param b  the open binder
 *  @param w  the family state: adapter fields and every layer's LoRA pairs are written */
void bind_adapter(const pulsar_model *m, pulsar_role_binder *b, pulsar_gemma4_model *w) {
    pulsar_str kind = {NULL, 0};
    if (!model_get_string(m, "pulsar.adapter", &kind)) return;
    const pulsar_gemma4_shape *s = &w->shape;
    float alpha = 0.0f;
    if (!pulsar_streq(kind, "jev") || !model_get_u32(m, "pulsar.adapter.lora_r", &w->lora_r) ||
        !model_get_f32_compat(m, "pulsar.adapter.lora_alpha", &alpha) || w->lora_r == 0 ||
        w->lora_r > PULSAR_GEMMA4_LORA_RANK_MAX || !(alpha > 0.0f)) {
        fprintf(stderr, "pulsar: %s: the container's adapter is '%.*s', rank %u, alpha %g: the runtime adapter takes a "
                        "jev LoRA of rank 1..%u and a positive alpha -- refusing\n", OWNER, (int)kind.len, kind.ptr,
                w->lora_r, (double)alpha, PULSAR_GEMMA4_LORA_RANK_MAX);
        pulsar_load_refuse();
        return;
    }
    w->adapter = true;
    w->lora_scale = alpha / (float)w->lora_r;
    const uint64_t r = w->lora_r, E = s->n_embd, nq = s->n_head;
    for (uint32_t il = 0; il < s->n_layer; il++) {
        pulsar_gemma4_block *L = &w->layer[il];
        const bool full = pulsar_gemma4_is_full(s, il);
        const uint64_t nkv = full ? s->n_head_kv : s->swa_n_head_kv, hd = full ? s->head_dim : s->swa_head_dim;
        bind_lora(b, "attn_q", (int32_t)il, r, E, nq * hd, &L->lora_q);
        bind_lora(b, "attn_k", (int32_t)il, r, E, nkv * hd, &L->lora_k);
        if (!full) bind_lora(b, "attn_v", (int32_t)il, r, E, nkv * hd, &L->lora_v);
        bind_lora(b, "attn_o", (int32_t)il, r, nq * hd, E, &L->lora_o);
        bind_lora(b, "ffn_gate", (int32_t)il, r, E, s->n_ff, &L->lora_gate);
        bind_lora(b, "ffn_up", (int32_t)il, r, E, s->n_ff, &L->lora_up);
        bind_lora(b, "ffn_down", (int32_t)il, r, s->n_ff, E, &L->lora_down);
    }
    const pulsar_tensor *h = pulsar_role_binder_find(b, pulsar_role_top("decision_head"));
    w->decision_slots = h && h->ndim == 2 ? (uint32_t)h->dim[1] : 0;
    if (!w->decision_slots) {
        fprintf(stderr, "pulsar: %s: the adapter carries no [slots][%u] decision_head -- refusing\n", OWNER, s->n_embd);
        pulsar_load_refuse();
        return;
    }
    w->decision_head = pulsar_role_bind(b, pulsar_role_top("decision_head"), A_F32, 2, E, w->decision_slots);
    w->decision_head_bias = pulsar_role_bind(b, pulsar_role_top("decision_head_bias"), A_F32, 1, w->decision_slots);
}

/** One layer: attention at the layer kind's geometry (a global layer has no attn_v), the dense MLP, the MoE.
 *  @param b   the open binder
 *  @param s   the shape (widths)
 *  @param il  the layer
 *  @param L   receives the layer's tensors */
void bind_layer(pulsar_role_binder *b, const pulsar_gemma4_shape *s, int32_t il, pulsar_gemma4_block *L) {
    const bool full = pulsar_gemma4_is_full(s, (uint32_t)il);
    const uint64_t E = s->n_embd, nq = s->n_head;
    const uint64_t nkv = full ? s->n_head_kv : s->swa_n_head_kv, hd = full ? s->head_dim : s->swa_head_dim;
    auto k = [&](const char *role) { return pulsar_role_in(role, LAYERS, il); };
    L->attn_norm = pulsar_role_bind(b, k("attn_norm"), A_BF16, 1, E);
    L->attn_q = pulsar_role_bind(b, k("attn_q"), A_LINEAR, 2, E, nq * hd);
    L->attn_k = pulsar_role_bind(b, k("attn_k"), A_LINEAR, 2, E, nkv * hd);
    if (!full) L->attn_v = pulsar_role_bind(b, k("attn_v"), A_LINEAR, 2, E, nkv * hd);   /* K = V on a global layer */
    L->attn_o = pulsar_role_bind(b, k("attn_o"), A_LINEAR, 2, nq * hd, E);
    L->attn_q_norm = pulsar_role_bind(b, k("attn_q_norm"), A_BF16, 1, hd);
    L->attn_k_norm = pulsar_role_bind(b, k("attn_k_norm"), A_BF16, 1, hd);
    L->attn_post_norm = pulsar_role_bind(b, k("attn_post_norm"), A_BF16, 1, E);
    /* the dense MLP, at its stored width */
    const uint64_t F = pulsar_gemma4_ff_stored(s);
    L->ffn_pre_norm = pulsar_role_bind(b, k("ffn_pre_norm"), A_BF16, 1, E);
    L->ffn_gate = pulsar_role_bind(b, k("ffn_gate"), A_LINEAR, 2, E, F);
    L->ffn_up = pulsar_role_bind(b, k("ffn_up"), A_LINEAR, 2, E, F);
    L->ffn_down = pulsar_role_bind(b, k("ffn_down"), A_LINEAR, 2, F, E);
    L->ffn_post_norm_1 = pulsar_role_bind(b, k("ffn_post_norm_1"), A_BF16, 1, E);
    /* the MoE, its experts at their stored width */
    const uint64_t Fe = pulsar_gemma4_ff_exp_stored(s), X = s->n_expert;
    L->moe_pre_norm = pulsar_role_bind(b, k("moe_pre_norm"), A_BF16, 1, E);
    L->router = pulsar_role_bind(b, k("router"), A_BF16, 2, E, X);
    L->router_scale = pulsar_role_bind(b, k("router_scale"), A_BF16, 1, E);
    L->router_expert_scale = pulsar_role_bind(b, k("router_expert_scale"), A_BF16, 1, X);
    L->exp_gate = pulsar_role_bind(b, k("expert_gate"), A_EXP_GATE_UP, 3, E, Fe, X);
    L->exp_up = pulsar_role_bind(b, k("expert_up"), A_EXP_GATE_UP, 3, E, Fe, X);
    L->exp_down = pulsar_role_bind(b, k("expert_down"), A_EXP_DOWN, 3, Fe, E, X);
    if (L->exp_gate && L->exp_up && L->exp_down &&
        !pulsar_format_moe_combo(L->exp_gate, L->exp_up, L->exp_down, PULSAR_ACT_ROWS_BF16, OWNER))
        pulsar_load_refuse();
    L->ffn_post_norm_2 = pulsar_role_bind(b, k("ffn_post_norm_2"), A_BF16, 1, E);
    L->ffn_post_norm = pulsar_role_bind(b, k("ffn_post_norm"), A_BF16, 1, E);
    L->layer_scalar = pulsar_role_bind(b, k("layer_scalar"), A_BF16, 1, 1);
}

}  // namespace

/** @param m  the container
 *  @param w  the family state: its shape is read; every tensor pointer and the census are written
 *  @return true = bound; false = refused (said) */
bool pulsar_gemma4_bind(const pulsar_model *m, pulsar_gemma4_model *w) {
    const pulsar_gemma4_shape *s = &w->shape;
    pulsar_role_binder *b = pulsar_role_binder_open(m, OWNER);
    w->embed = pulsar_role_bind(b, pulsar_role_top("embed"), A_BF16, 2, s->n_embd, s->n_vocab);
    w->output_norm = pulsar_role_bind(b, pulsar_role_top("output_norm"), A_BF16, 1, s->n_embd);
    w->head = pulsar_role_bind(b, pulsar_role_top("head"), A_LINEAR, 2, s->n_embd, s->n_vocab);
    for (uint32_t il = 0; il < s->n_layer; il++) bind_layer(b, s, (int32_t)il, &w->layer[il]);
    bind_adapter(m, b, w);
    /* L338 P3: the block drafter (block `drafter`) when the open loaded it, else unloaded */
    if (w->block) {
        (void)pulsar_block_drafter_bind(b, w->block);
    } else if (w->block_written) {
        const uint64_t n = pulsar_role_binder_unload(b, "drafter");
        fprintf(stderr, "pulsar: %s: the block drafter (%llu tensors) is not loaded (--drafter none)\n", OWNER,
                (unsigned long long)n);
    }
    /* the census: every directory entry declared, every byte priced (the extras refused by name) */
    return pulsar_role_binder_close(b, &w->census);
}

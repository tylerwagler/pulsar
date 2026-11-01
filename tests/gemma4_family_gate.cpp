/* gemma4_family_gate.cpp -- L338 E2 / E4: the Gemma 4 family on a container, graded on what it claims today.
 *
 *   gemma4_family_gate MODEL
 *
 * Opens MODEL inspect-only (no GPU, no weight byte staged) and checks the family's load: the family is chosen by
 * `general.architecture`; the plan is the config's (window vs full attention from layer_types, the dense MLP + MoE on
 * every layer); EVERY tensor of the directory is bound by its bind record, every byte priced by a role, and the price is
 * what the core stages and admission charges; every slot the plan declares is bound (a global layer has no attn_v: K =
 * V), the dense MLP and the experts at the pack's stored widths; the engine facts; and the session side as declared
 * (E4: the core driver's table covers the plan, the window ring + global pool state, the caps).  The forward itself is
 * graded on the GPU by the core driver's forward gate + tests/gemma4_grade.py.  Exit 0 = every check passed. */
#include "families/gemma4/gemma4.h"
#include "engine/exl3_trellis.h"

#include <stdarg.h>

static int n_fail;

static void check(bool ok, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "  %s ", ok ? "ok  " : "FAIL");
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    if (!ok) n_fail++;
}

/* every slot of a layer bound, attn_v exactly on a sliding layer */
static bool layer_bound(const pulsar_gemma4_block &L, bool full) {
    return L.attn_norm && L.attn_q && L.attn_k && (L.attn_v != NULL) == !full && L.attn_o && L.attn_q_norm &&
           L.attn_k_norm && L.attn_post_norm && L.ffn_pre_norm && L.ffn_gate && L.ffn_up && L.ffn_down &&
           L.ffn_post_norm_1 && L.moe_pre_norm && L.router && L.router_scale && L.router_expert_scale && L.exp_gate &&
           L.exp_up && L.exp_down && L.ffn_post_norm_2 && L.ffn_post_norm && L.layer_scalar;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL\n", argv[0]);
        return 2;
    }
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    opt.inspect_only = true;
    pulsar_engine *e = NULL;
    fprintf(stderr, "gemma4-family gate: open %s\n", argv[1]);
    const int rc = pulsar_engine_open(&e, &opt);
    check(rc == 0 && e, "the container opens (--inspect)");
    if (rc != 0 || !e) return 1;
    const pulsar_gemma4_model *w = pulsar_gemma4_of(e);
    check(pulsar_engine_family(e) == PULSAR_FAMILY_ID_GEMMA4 && w, "family is gemma4 (%s)", pulsar_engine_family_name(e));
    if (!w) return 1;
    const pulsar_gemma4_shape *s = &w->shape;
    const pulsar_layer_plan *p = &e->plan;

    /* the plan is the config's */
    bool plan_ok = p->n_layer == s->n_layer && s->n_layer > 0;
    uint32_t n_full = 0;
    for (uint32_t il = 0; plan_ok && il < p->n_layer; il++) {
        const pulsar_layer_desc &d = p->desc[il];
        const bool full = pulsar_gemma4_is_full(s, il);
        plan_ok = p->kind[il] == PULSAR_LAYER_BLOCKS &&
                  d.mixer == (full ? PULSAR_MIXER_ATTN_FULL : PULSAR_MIXER_ATTN_WINDOW) &&
                  d.window == (full ? 0u : s->window) && d.ffn == PULSAR_FFN_DENSE_MOE;
        n_full += full;
    }
    check(plan_ok, "plan: %u layers (%s), %u window (%u keys) + %u full, every layer dense MLP + MoE, from layer_types",
          p->n_layer, pulsar_layer_kind_name(p->kind[0]), p->n_layer - n_full, s->window, n_full);
    uint32_t at = 0;
    check(pulsar_layer_plan_check(p, &at) == NULL, "the plan's invariants hold (PULSAR_FFN_DENSE_MOE admitted)");

    /* every tensor bound, every byte priced, the price is what is staged and charged */
    uint64_t dir_bytes = 0, roleless = 0;
    for (uint64_t i = 0; i < e->model.n_tensors; i++) {
        dir_bytes += e->model.tensors[i].bytes;
        roleless += e->model.tensors[i].bind_role.len == 0;
    }
    const pulsar_role_census &c = w->census;
    check(roleless == 0, "every directory entry carries a bind record (%llu without)", (unsigned long long)roleless);
    check(c.tensors == e->model.n_tensors, "every tensor bound: %llu of %llu, in %u roles", (unsigned long long)c.tensors,
          (unsigned long long)e->model.n_tensors, c.roles);
    check(c.bytes == dir_bytes && c.directory_bytes == dir_bytes,
          "every byte priced by a role: %llu of the directory's %llu", (unsigned long long)c.bytes,
          (unsigned long long)dir_bytes);
    check(pulsar_engine_weights_resident_bytes(e) == e->model.mapped_bytes && c.bytes <= e->model.mapped_bytes,
          "admission charges the staged container (%.3f GiB mapped; the bound tensors %.3f GiB, the rest headers)",
          (double)e->model.mapped_bytes / 1073741824.0, (double)c.bytes / 1073741824.0);

    /* every slot the plan declares is bound; the stored widths */
    bool slots = w->embed && w->output_norm && w->head;
    for (uint32_t il = 0; slots && il < s->n_layer; il++) slots = layer_bound(w->layer[il], pulsar_gemma4_is_full(s, il));
    check(slots, "every slot bound: embed, output_norm, head (the pack's EXL3), and each layer's (no attn_v on a global "
                 "layer)");
    check(exl3_type_k2(w->head->type) > 0, "the head is EXL3 (rate K%d: the Turing build's head)", exl3_type_k2(w->head->type) / 2);
    const pulsar_gemma4_block &L0 = w->layer[0];
    check(pulsar_gemma4_ff_stored(s) == 2176 && pulsar_gemma4_ff_exp_stored(s) == 768 &&
              L0.ffn_down->dim[0] == 2176 && L0.exp_down->dim[0] == 768 && L0.exp_gate->dim[1] == 768 &&
              L0.exp_gate->dim[2] == s->n_expert,
          "the dense MLP (%u -> stored %u) and the experts (%u -> stored %u) bound at the pack's stored widths",
          s->n_ff, pulsar_gemma4_ff_stored(s), s->n_ff_exp, pulsar_gemma4_ff_exp_stored(s));
    check(!pulsar_engine_has_vision(e), "images are refused (text only)");

    /* the engine facts */
    check(pulsar_engine_logits_width(e) == (int)PULSAR_GEMMA4_SHAPE_26B_A4B.n_vocab, "logits width %d",
          pulsar_engine_logits_width(e));
    check(!strcmp(pulsar_engine_model_name(e), "Gemma-4-26B-A4B"), "model name '%s'", pulsar_engine_model_name(e));
    check(pulsar_engine_model_id(e) == 7, "disk-KV model id %d", pulsar_engine_model_id(e));
    check(!pulsar_engine_has_tokenizer(e) && pulsar_engine_chat_format(e) == PULSAR_CHAT_NONE,
          "no chat front yet (no CAP_CHAT)");
    check(pulsar_engine_drafter(e) == PULSAR_DRAFTER_NONE, "no drafter runs");
    int k2 = 0;
    for (uint32_t il = 0; il < s->n_layer; il++) {
        const int g = exl3_type_k2(w->layer[il].exp_gate->type);
        if (g > k2) k2 = g;
    }
    check(k2 > 0 && e->routed_quant_bits() == 20 + k2, "routed quant tier %d (EXL3: 20 + the highest half-bit rate)",
          e->routed_quant_bits());

    /* the session side as declared (E4): the driver's table covers the plan; the state is a window layer's K|V ring
     * (RING over the 1024-key window, 8 kv x 256 BF16) and a global layer's K|V pool (POOL, one token a row, 2 kv x
     * 512 BF16); the session ops are the core driver's over the declared banks */
    uint32_t miss_at = 0;
    const char *miss = pulsar_driver_first_missing(e->family->driver, p, &miss_at);
    check(e->family->driver == &PULSAR_GEMMA4_DRIVER && !miss, "the core driver's table covers the plan (%s)",
          miss ? miss : "nothing missing");
    const pulsar_state_plan *sp = &e->state_plan;
    const pulsar_state_decl &win = sp->decl[PULSAR_GEMMA4_STATE_KV_WIN], &full = sp->decl[PULSAR_GEMMA4_STATE_KV_FULL];
    bool state_ok = sp->n_decl == PULSAR_GEMMA4_STATE_COUNT && sp->split_invariant && sp->resume_grid > 0 &&
                    win.kind == PULSAR_STATE_RING && win.capture == PULSAR_STATE_CAPTURE_NONE && win.tokens == s->window &&
                    win.bytes == 2ull * s->swa_n_head_kv * s->swa_head_dim * 2u &&
                    full.kind == PULSAR_STATE_POOL && full.capture == PULSAR_STATE_CAPTURE_NONE && full.tokens == 1u &&
                    full.bytes == 2ull * s->n_head_kv * s->head_dim * 2u &&
                    sp->n_trunk == s->n_layer && sp->n_slots == s->n_layer;
    for (uint32_t il = 0; state_ok && il < s->n_layer; il++)
        state_ok = sp->slot[il] == 1u << (pulsar_gemma4_is_full(s, il) ? PULSAR_GEMMA4_STATE_KV_FULL
                                                                        : PULSAR_GEMMA4_STATE_KV_WIN);
    check(state_ok, "the declared state: kv_win RING %u keys x %llu B a window layer, kv_full POOL %llu B a token a "
          "global layer", win.tokens, (unsigned long long)win.bytes, (unsigned long long)full.bytes);
    check(e->family->caps == (PULSAR_FAMILY_CAP_BANKS | PULSAR_FAMILY_CAP_SEGMENTS | PULSAR_FAMILY_CAP_PAYLOAD) &&
              e->family->banks == &PULSAR_BANK_OPS_DECLARED && e->family->spec == NULL &&
              e->family->driver->absorb == NULL && e->family->driver->mtp == NULL,
          "caps 0x%x (banks, segments, payload: the driver's session over the declared state; no chat, spec, TP or "
          "images yet)", e->family->caps);
    pulsar_engine_close(e);
    fprintf(stderr, "GEMMA4-FAMILY PROBE %s\n", n_fail ? "FAIL" : "PASS");
    return n_fail ? 1 : 0;
}

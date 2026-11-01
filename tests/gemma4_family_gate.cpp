/* gemma4_family_gate.cpp -- L338 E2: the Gemma 4 family on a container, graded on what it claims today.
 *
 *   gemma4_family_gate MODEL
 *
 * Opens MODEL inspect-only (no GPU, no weight byte staged) and checks the family's load: the family is chosen by
 * `general.architecture`; the plan is the config's (window vs full attention from layer_types, the dense MLP + MoE on
 * every layer); EVERY tensor of the directory is bound by its bind record, every byte priced by a role, and the price is
 * what the core stages and admission charges; every slot the plan declares is bound (a global layer has no attn_v: K =
 * V), the dense MLP and the experts at the pack's stored widths; the engine facts; and the session entries refuse by
 * name (Gemma 4's blocks on the core driver are not written yet).  Exit 0 = every check passed. */
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
    bool slots = w->embed && w->output_norm;
    for (uint32_t il = 0; slots && il < s->n_layer; il++) slots = layer_bound(w->layer[il], pulsar_gemma4_is_full(s, il));
    check(slots, "every slot bound: embed (= the tied head), output_norm, and each layer's (no attn_v on a global layer)");
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

    /* the session entries refuse by name: Gemma 4's driver blocks are not written */
    pulsar_session *dummy = (pulsar_session *)xcalloc(1, sizeof(pulsar_session));
    dummy->engine = e;
    const pulsar_family_session_ops *so = e->family->session;
    check(so->create(dummy, 1) != 0, "a session's create refuses (the log names the driver blocks)");
    char err[512] = "";
    check(so->eval_row(dummy, 0, err, sizeof(err)) != 0 && strstr(err, "driver blocks"), "eval_row refuses: %s", err);
    err[0] = 0;
    check(so->decode_multiseq(dummy, NULL, 0, NULL, 0, err, sizeof(err)) != 0 && strstr(err, "driver blocks"),
          "decode_multiseq refuses by name");
    err[0] = 0;
    check(so->decode_mixed(dummy, NULL, 0, NULL, 0, NULL, 0, err, sizeof(err)) != 0 && strstr(err, "driver blocks"),
          "decode_mixed refuses by name");
    free(dummy);
    pulsar_engine_close(e);
    fprintf(stderr, "GEMMA4-FAMILY PROBE %s\n", n_fail ? "FAIL" : "PASS");
    return n_fail ? 1 : 0;
}

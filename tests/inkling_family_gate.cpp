/* inkling_family_gate.cpp -- L295 E1: the Inkling family on a container, graded on what it claims today.
 *
 *   inkling_family_gate MODEL
 *
 * Opens MODEL inspect-only (no GPU, no weight byte staged) and checks the family's load: the family is chosen by
 * `general.architecture`; the plan is the config's (window vs full attention, dense vs MoE per layer); EVERY tensor
 * of the directory is bound by its bind record, every byte priced by a role, and the price is what the core stages
 * and admission charges; every slot the plan declares is bound; the engine facts; the tokenizer and chat front (L294
 * E6: the container's tokenizer files, F5's renderer turn by turn); and the session entries refuse by
 * name (Inkling's blocks on F4's core driver are not written yet).  The refusals (a missing tensor, a wrong shape, an unknown switch, ...) are graded by
 * tests/inkling_family_gate.py, which runs this probe on the clean container and tests/loader_open on each mutant.
 * Exit 0 = every check passed. */
#include "families/inkling/inkling.h"
#include "engine/exl3_trellis.h"

#include <stdarg.h>

#include <string>

static int n_fail;

/* an added token of the renderer's spelling: <|...|> as one id */
static bool piece_is_marker(pulsar_engine *e, int id) {
    size_t n = 0;
    char *p = pulsar_token_text(e, id, &n);
    const bool m = n > 4 && !strncmp(p, "<|", 2) && !strcmp(p + n - 2, "|>");
    free(p);
    return m;
}

static void check(bool ok, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "  %s ", ok ? "ok  " : "FAIL");
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    if (!ok) n_fail++;
}

static bool block_bound(const pulsar_inkling_block &L, const pulsar_layer_desc &d, uint32_t n_shared) {
    bool ok = L.attn_norm && L.attn_q && L.attn_k && L.attn_v && L.attn_rel_proj && L.attn_o && L.attn_q_norm &&
              L.attn_k_norm && L.attn_rel_basis && L.conv_k && L.conv_v && L.conv_attn_out && L.conv_ffn_out && L.ffn_norm;
    if (d.ffn == PULSAR_FFN_DENSE)
        return ok && L.ffn_gate && L.ffn_up && L.ffn_down && L.ffn_out_scale && !L.router && !L.exp_gate;
    ok = ok && L.router && L.router_select_bias && L.router_out_scale && L.exp_gate && L.exp_up && L.exp_down && !L.ffn_gate;
    for (uint32_t j = 0; j < n_shared; j++) ok = ok && L.sh_gate[j] && L.sh_up[j] && L.sh_down[j];
    return ok;
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
    fprintf(stderr, "inkling-family gate: open %s\n", argv[1]);
    const int rc = pulsar_engine_open(&e, &opt);
    check(rc == 0 && e, "the container opens (--inspect)");
    if (rc != 0 || !e) return 1;
    const pulsar_inkling_model *w = pulsar_inkling_of(e);
    check(pulsar_engine_family(e) == PULSAR_FAMILY_ID_INKLING && w, "family is inkling (%s)", pulsar_engine_family_name(e));
    if (!w) return 1;
    const pulsar_inkling_shape *s = &w->shape;
    const pulsar_layer_plan *p = &e->plan;

    /* the plan is the config's */
    bool plan_ok = p->n_layer == s->n_layer && s->n_layer > 0;
    uint32_t n_win = 0, n_moe = 0;
    for (uint32_t il = 0; plan_ok && il < p->n_layer; il++) {
        const pulsar_layer_desc &d = p->desc[il];
        const bool window = s->window_layers >> il & 1u;
        plan_ok = p->kind[il] == PULSAR_LAYER_BLOCKS &&
                  d.mixer == (window ? PULSAR_MIXER_ATTN_WINDOW : PULSAR_MIXER_ATTN_FULL) &&
                  d.window == (window ? s->window : 0u) && d.ffn == (il < s->n_dense ? PULSAR_FFN_DENSE : PULSAR_FFN_MOE);
        n_win += window;
        n_moe += d.ffn == PULSAR_FFN_MOE;
    }
    check(plan_ok, "plan: %u layers (%s), %u window + %u full, %u dense + %u MoE, from local_layer_ids / dense_mlp_idx",
          p->n_layer, pulsar_layer_kind_name(p->kind[0]), n_win, p->n_layer - n_win, p->n_layer - n_moe, n_moe);
    check(pulsar_layer_plan_count(p, PULSAR_LAYER_BLOCKS) == p->n_layer, "every layer is a PULSAR_LAYER_BLOCKS layer");

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

    /* every slot the plan declares is bound */
    bool slots = w->embed && w->embed_norm && w->output_norm && w->head;
    for (uint32_t il = 0; slots && il < s->n_layer; il++) slots = block_bound(w->layer[il], p->desc[il], s->n_shared);
    check(slots, "every slot bound: embed, embed_norm, output_norm, head, and each layer's blocks");
    if (s->mtp_written) {
        bool mtp = s->n_mtp > 0;
        for (uint32_t n = 0; mtp && n < s->n_mtp; n++) {
            const pulsar_layer_desc dense = {PULSAR_MIXER_NONE, PULSAR_FFN_DENSE, 0};
            mtp = block_bound(w->mtp[n].block, dense, 0) && w->mtp[n].embed_norm && w->mtp[n].hidden_norm && w->mtp[n].in_proj;
        }
        check(mtp, "the drafter's %u MTP layers are bound", s->n_mtp);
    }
    if (s->vision_written) {
        bool vis = w->vis_final_norm != NULL;
        for (uint32_t i = 0; vis && i < s->vis_layers; i++) vis = w->vis_linear[i] && (i + 1 == s->vis_layers || w->vis_norm[i]);
        check(vis, "the hmlp tower's %u linears are bound", s->vis_layers);
    }
    check(!pulsar_engine_has_vision(e), "images are refused (no image front until E9)");

    /* the engine facts */
    check(pulsar_engine_logits_width(e) == (int)PULSAR_INKLING_SHAPE_SMALL.n_vocab, "logits width %d",
          pulsar_engine_logits_width(e));
    check(!strcmp(pulsar_engine_model_name(e), "Inkling-Small"), "model name '%s'", pulsar_engine_model_name(e));
    check(pulsar_engine_model_id(e) == 3, "disk-KV model id %d (0/1 DeepSeek, 2 Qwen)", pulsar_engine_model_id(e));
    /* the tokenizer and chat front (E6): F5's BPE engine on the container's tokenizer files, F5's renderer */
    check(pulsar_engine_has_tokenizer(e) && pulsar_engine_chat_format(e) == PULSAR_CHAT_INKLING,
          "the tokenizer + chat front (CAP_CHAT, the Inkling chat format)");
    check(pulsar_engine_vocab_size(e) > 0 && (uint32_t)pulsar_engine_vocab_size(e) <= s->n_vocab_real,
          "the tokenizer table: %d ids (<= the head's %u real rows)", pulsar_engine_vocab_size(e), s->n_vocab_real);
    check(pulsar_token_eos(e) == (int)s->eos_id && pulsar_token_is_stop(e, (int)s->eos_id),
          "eos %d is the container's eos_token_id %u", pulsar_token_eos(e), s->eos_id);
    check(pulsar_engine_think_default(e) == PULSAR_THINK_DEFAULT &&
              pulsar_engine_think_mode_supported(e, PULSAR_THINK_NONE, NULL, 0) &&
              !pulsar_engine_think_mode_supported(e, PULSAR_THINK_LOW, NULL, 0),
          "thinking on at the template's default effort, or off");
    {
        pulsar_turn_markers m = {};
        pulsar_tokens t = {0};
        pulsar_encode_chat_prompt(e, "Be terse.", "Hi <|end_message|>", PULSAR_THINK_DEFAULT, &t);
        pulsar_chat_end_assistant(e, &t);
        std::string text;
        for (int i = 0; i < t.len; i++) {
            size_t n = 0;
            char *piece = pulsar_token_text(e, t.v[i], &n);
            text.append(piece, n);
            free(piece);
        }
        const char *want = "<|message_system|><|content_text|>Be terse.<|end_message|><|message_system|><|content_text|>"
                           "Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hi <|end_message|>"
                           "<|end_message|><|message_model|><|content_model_end_sampling|>";
        check(text == want && t.len > 0 && t.v[t.len - 1] == (int)s->eos_id,
              "the chat front: head + turn + close == the renderer's text, closed by eos (%d tokens)", t.len);
        int markers = 0;   /* the client's spelling of a marker is text (L223): only the template's markers count */
        for (int i = 0; i < t.len; i++) markers += t.v[i] != pulsar_token_eos(e) && piece_is_marker(e, t.v[i]);
        check(markers == 10, "%d marker tokens (the template's ten; the client's \"<|end_message|>\" stays text)", markers);
        check(pulsar_chat_turn_markers(e, &m) && m.n_user == 1 && m.n_assistant == 1 &&
                  pulsar_turn_marker_at(&m, t.v, t.len, 0) == 0,
              "turn markers: <|message_user|> / <|message_model|>, one token each");
        pulsar_tokens_free(&t);
    }
    check(pulsar_engine_drafter(e) == PULSAR_DRAFTER_NONE, "no drafter runs (F6)");
    int k2 = 0;
    for (uint32_t il = s->n_dense; il < s->n_layer; il++) {
        const int g = exl3_type_k2(w->layer[il].exp_gate->type);
        if (g > k2) k2 = g;
    }
    check(e->routed_quant_bits() == (k2 ? 20 + k2 : 0), "routed quant tier %d (EXL3: 20 + the highest half-bit rate)",
          e->routed_quant_bits());

    /* the session entries refuse by name: Inkling's driver blocks are not written (F4's driver is in) */
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
    fprintf(stderr, "INKLING-FAMILY PROBE %s\n", n_fail ? "FAIL" : "PASS");
    return n_fail ? 1 : 0;
}

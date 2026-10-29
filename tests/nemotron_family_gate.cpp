/* nemotron_family_gate.cpp -- L296 N1: the Nemotron-H family on a container, graded on what it claims today.
 *
 *   nemotron_family_gate MODEL
 *
 * Opens MODEL inspect-only (no GPU, no weight byte staged) and checks the family's load: the family is chosen by
 * `general.architecture`; the plan is the config's (the HF blocks grouped: {Mamba-2 | attention} x {MoE | none});
 * EVERY tensor of the directory is bound by its bind record, every byte priced by a role, and the price is what the
 * core stages and admission charges; every slot the plan declares is bound (and none it does not); the drafter's;
 * the engine facts; the tokenizer and chat front (N8: the container's tokenizer.json, the renderer turn by turn); and
 * the session side as declared (N3 / N7) -- the core driver's table covers the plan, the declared state is SSM FIXED
 * DELTA + conv FIXED SNAPSHOT on each Mamba-2 layer and a KV POOL on each attention layer, the drafter's slot (its KV
 * POOL and the parked row, the chained-MTP lane's hooks) when the container holds the MTP module, the drafter, the
 * caps (the forward itself is tests/nemotron_fwd_gate's, on the GPU).  The refusals (a missing tensor, a wrong
 * shape, an unknown switch, ...) are graded by tests/nemotron_family_gate.py, which runs this probe on the clean
 * container and tests/loader_open on each mutant.  Exit 0 = every check passed. */
#include "families/nemotron_h/nemotron_h.h"
#include "engine/exl3_trellis.h"
#include "engine/spec_block.h"
#include "engine/spec_mtp.h"

#include <stdarg.h>

#include <string>

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

/* the slots a layer's kinds declare are bound, and the others are not */
static bool block_bound(const pulsar_nemotron_h_block &L, const pulsar_layer_desc &d) {
    const bool mamba = d.mixer == PULSAR_MIXER_MAMBA2, moe = d.ffn == PULSAR_FFN_MOE;
    const bool ssm = L.ssm_pre_norm && L.ssm_in && L.ssm_conv && L.ssm_conv_bias && L.ssm_dt_bias && L.ssm_a_log &&
                     L.ssm_d && L.ssm_out_norm && L.ssm_out;
    const bool no_ssm = !L.ssm_pre_norm && !L.ssm_in && !L.ssm_out;
    const bool attn = L.attn_norm && L.attn_q && L.attn_k && L.attn_v && L.attn_o;
    const bool no_attn = !L.attn_norm && !L.attn_q && !L.attn_o;
    const bool ffn = L.ffn_norm && L.router && L.router_select_bias && L.latent_in && L.latent_out && L.exp_up &&
                     L.exp_down && L.sh_up && L.sh_down;
    const bool no_ffn = !L.ffn_norm && !L.router && !L.exp_up;
    return (mamba ? ssm && no_attn : attn && no_ssm) && (moe ? ffn : no_ffn);
}

static std::string tokens_text(pulsar_engine *e, const pulsar_tokens &t) {
    std::string text;
    for (int i = 0; i < t.len; i++) {
        size_t n = 0;
        char *piece = pulsar_token_text(e, t.v[i], &n);
        text.append(piece, n);
        free(piece);
    }
    return text;
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
    fprintf(stderr, "nemotron-family gate: open %s\n", argv[1]);
    const int rc = pulsar_engine_open(&e, &opt);
    check(rc == 0 && e, "the container opens (--inspect)");
    if (rc != 0 || !e) return 1;
    const pulsar_nemotron_h_model *w = pulsar_nemotron_h_of(e);
    check(pulsar_engine_family(e) == PULSAR_FAMILY_ID_NEMOTRON_H && w, "family is nemotron_h (%s)",
          pulsar_engine_family_name(e));
    if (!w) return 1;
    const pulsar_nemotron_h_shape *s = &w->shape;
    const pulsar_layer_plan *p = &e->plan;

    /* the plan is the config's grouping */
    bool plan_ok = p->n_layer == s->n_layer && s->n_layer > 0;
    uint32_t n_mamba = 0, n_moe = 0;
    std::string kinds;
    for (uint32_t il = 0; plan_ok && il < p->n_layer; il++) {
        const pulsar_layer_desc &d = p->desc[il];
        const bool mamba = s->mamba_layers >> il & 1u, moe = s->moe_layers >> il & 1u;
        plan_ok = p->kind[il] == PULSAR_LAYER_BLOCKS && d.mixer == (mamba ? PULSAR_MIXER_MAMBA2 : PULSAR_MIXER_ATTN_FULL) &&
                  d.ffn == (moe ? PULSAR_FFN_MOE : PULSAR_FFN_NONE) && d.window == 0;
        n_mamba += mamba;
        n_moe += moe;
        kinds += mamba ? "M" : "*";
        kinds += moe ? "E" : "-";
    }
    check(plan_ok && n_mamba + n_moe + (s->n_layer - n_mamba) == s->n_blocks,
          "plan: %u blocks -> %u layers [%s]: %u Mamba-2 + %u attention, %u MoE + %u mixer-only", s->n_blocks,
          p->n_layer, kinds.c_str(), n_mamba, p->n_layer - n_mamba, n_moe, p->n_layer - n_moe);
    uint32_t at = 0;
    check(pulsar_layer_plan_check(p, &at) == NULL, "the plan passes the one layer-description check");

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
    check(c.bytes == dir_bytes && c.directory_bytes == dir_bytes, "every byte priced by a role: %llu of the directory's %llu",
          (unsigned long long)c.bytes, (unsigned long long)dir_bytes);
    check(pulsar_engine_weights_resident_bytes(e) == e->model.mapped_bytes && c.bytes <= e->model.mapped_bytes,
          "admission charges the staged container (%.3f GiB mapped; the bound tensors %.3f GiB, the rest headers)",
          (double)e->model.mapped_bytes / 1073741824.0, (double)c.bytes / 1073741824.0);

    /* every slot the plan declares is bound, and only those */
    bool slots = w->embed && w->output_norm && w->head;
    for (uint32_t il = 0; slots && il < s->n_layer; il++) slots = block_bound(w->layer[il], p->desc[il]);
    check(slots, "every slot bound: embed, output_norm, head, and each layer's mixer / MoE slots (none of the other kind)");
    if (s->mtp_written) {
        const pulsar_layer_desc attn_moe = {PULSAR_MIXER_ATTN_FULL, PULSAR_FFN_MOE, 0};
        const pulsar_nemotron_h_draft &D = w->mtp;
        check(block_bound(D.block, attn_moe) && D.embed_norm && D.hidden_norm && D.in_proj && D.out_norm,
              "the drafter's {attention, MoE} layer, its input side and its out norm are bound");
    }
    check(pulsar_engine_has_vision(e) == s->vision_written,
          "images served exactly when the container holds the tower (L331; vision_present %d)", (int)s->vision_written);

    /* the engine facts */
    check(pulsar_engine_logits_width(e) == (int)PULSAR_NEMOTRON_H_SHAPE_SUPER.n_vocab, "logits width %d",
          pulsar_engine_logits_width(e));
    check(!strcmp(pulsar_engine_model_name(e), PULSAR_NEMOTRON_H_SHAPE_SUPER.name), "model name '%s'",
          pulsar_engine_model_name(e));
    check(pulsar_engine_model_id(e) == 4, "disk-KV model id %d (0/1 DeepSeek, 2 Qwen, 3 Inkling)", pulsar_engine_model_id(e));
    /* the tokenizer and chat front (N8) */
    check(pulsar_engine_has_tokenizer(e) && pulsar_engine_chat_format(e) == PULSAR_CHAT_NEMOTRON_H,
          "the tokenizer + chat front (CAP_CHAT, the Nemotron-H chat format)");
    check(pulsar_engine_vocab_size(e) == (int)s->n_vocab, "the tokenizer table: %d ids (== the head's rows)",
          pulsar_engine_vocab_size(e));
    check(pulsar_token_eos(e) == (int)s->eos_id && pulsar_token_is_stop(e, (int)s->eos_id) && pulsar_token_is_stop(e, 2),
          "eos %d is the container's eos_token_id %u; </s> 2 stops too (generation_config)", pulsar_token_eos(e), s->eos_id);
    check(pulsar_engine_think_default(e) == PULSAR_THINK_DEFAULT &&
              pulsar_engine_think_mode_supported(e, PULSAR_THINK_NONE, NULL, 0) &&
              !pulsar_engine_think_mode_supported(e, PULSAR_THINK_LOW, NULL, 0),
          "thinking on at the template's default effort, or off");
    {
        pulsar_tokens t = {0};
        pulsar_encode_chat_prompt(e, "Be terse.", "Hi <|im_end|>", PULSAR_THINK_DEFAULT, &t);
        const std::string on = tokens_text(e, t);
        const int n_on = t.len;
        int im_end = 0;   /* the client's "<|im_end|>" stays text (L223): only the template's markers count */
        for (int i = 0; i < t.len; i++) im_end += t.v[i] == (int)s->eos_id;
        pulsar_chat_end_assistant(e, &t);
        check(on == "<|im_start|>system\nBe terse.<|im_end|>\n<|im_start|>user\nHi <|im_end|><|im_end|>\n"
                    "<|im_start|>assistant\n<think>\n" &&
                  im_end == 2 && tokens_text(e, t) == on + "<|im_end|>\n",
              "the chat front: head + turn == the template's text (%d tokens, %d <|im_end|> markers: the client's stays "
              "text), closed by <|im_end|>\\n", n_on, im_end);
        pulsar_tokens_free(&t);
        pulsar_encode_chat_prompt(e, NULL, "Hi", PULSAR_THINK_NONE, &t);
        check(tokens_text(e, t) == "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n"
                                   "<|im_start|>assistant\n<think></think>",
              "thinking off: an empty system turn (the template always writes one) and the closed think block");
        pulsar_tokens_free(&t);
        pulsar_turn_markers m = {};
        pulsar_encode_chat_prompt(e, NULL, "Hi", PULSAR_THINK_DEFAULT, &t);
        int n_user = 0, n_asst = 0;
        const bool have = pulsar_chat_turn_markers(e, &m);
        for (int i = 0; have && i < t.len; i++) {
            const int r = pulsar_turn_marker_at(&m, t.v, t.len, i);
            n_user += r == 1;
            n_asst += r == 2;
        }
        check(have && m.n_user == 2 && m.n_assistant == 2 && n_user == 1 && n_asst == 1,
              "turn markers: <|im_start|> + the role word's first token (\"user\"; \"ass\" of \"ass\" \"istant\"); one "
              "user and one assistant turn found in a rendered prompt");
        pulsar_tokens_free(&t);
        check(pulsar_token_think_close(e) == 13, "the think close is </think> (13, a non-special added token)");
    }
    check(pulsar_engine_drafter(e) == (s->mtp_written ? PULSAR_DRAFTER_MTP : PULSAR_DRAFTER_NONE) &&
              e->drafter_ops == (s->mtp_written ? &PULSAR_NEMOTRON_H_DRAFTER : NULL) && w->drafting == s->mtp_written,
          "the drafter: %s (N9)", s->mtp_written ? "the MTP module, on the chained-MTP lane" : "none (no MTP module)");
    int k2 = 0;
    for (uint32_t il = 0; il < s->n_layer; il++)
        if (w->layer[il].exp_up) {
            const int g = exl3_type_k2(w->layer[il].exp_up->type);
            if (g > k2) k2 = g;
        }
    check(e->routed_quant_bits() == (k2 ? 20 + k2 : 0), "routed quant tier %d (EXL3: 20 + the highest half-bit rate; "
          "BF16: 0)", e->routed_quant_bits());

    /* the session side as declared (N3 / N7): the driver's table covers the plan; the state is the Mamba-2 layers'
     * SSM (FIXED, DELTA with its commit) + conv tail (FIXED, SNAPSHOT), the attention layers' K|V pool, the drafter's
     * slot's K|V pool when the container holds the MTP module */
    uint32_t miss_at = 0;
    const char *miss = pulsar_driver_first_missing(e->family->driver, p, &miss_at);
    check(e->family->driver && !miss, "the core driver's table covers the plan (%s)", miss ? miss : "nothing missing");
    const pulsar_state_plan *sp = &e->state_plan;
    const pulsar_state_decl &ssm = sp->decl[PULSAR_NEMOTRON_H_STATE_SSM], &conv = sp->decl[PULSAR_NEMOTRON_H_STATE_CONV],
                            &kv = sp->decl[PULSAR_NEMOTRON_H_STATE_KV];
    const uint64_t ssm_bytes = (uint64_t)s->ssm_n_head * s->ssm_head_dim * s->ssm_state * 4u;
    const uint64_t conv_ch = (uint64_t)s->ssm_n_head * s->ssm_head_dim + 2ull * s->ssm_groups * s->ssm_state;
    const uint64_t upd = ((uint64_t)s->ssm_n_head * s->ssm_head_dim + (uint64_t)s->ssm_groups * s->ssm_state +
                          2ull * s->ssm_n_head + 3u) / 4u * 4u * 4u;
    bool state_ok = sp->n_decl == PULSAR_NEMOTRON_H_STATE_COUNT && sp->split_invariant && sp->resume_grid > 0 &&
                    ssm.kind == PULSAR_STATE_FIXED && ssm.capture == PULSAR_STATE_CAPTURE_DELTA && ssm.commit &&
                    ssm.bytes == ssm_bytes && ssm.capture_bytes == upd &&
                    conv.kind == PULSAR_STATE_FIXED && conv.capture == PULSAR_STATE_CAPTURE_SNAPSHOT &&
                    conv.bytes == (uint64_t)(s->conv_k - 1u) * conv_ch * 2u &&
                    kv.kind == PULSAR_STATE_POOL && kv.tokens == 1u &&
                    kv.bytes == 2ull * s->n_head_kv * s->head_dim * 2u &&
                    sp->n_trunk == s->n_layer && sp->n_slots == s->n_layer + (s->mtp_written ? 1u : 0u);
    const pulsar_state_decl &pend = sp->decl[PULSAR_NEMOTRON_H_STATE_MTP_PEND];
    state_ok = state_ok && pend.kind == PULSAR_STATE_FIXED && pend.capture == PULSAR_STATE_CAPTURE_NONE && !pend.host &&
               pend.bytes == (uint64_t)s->n_embd * 2u;
    for (uint32_t il = 0; state_ok && il < s->n_layer; il++)
        state_ok = sp->slot[il] == (p->desc[il].mixer == PULSAR_MIXER_MAMBA2
                                        ? (1u << PULSAR_NEMOTRON_H_STATE_SSM | 1u << PULSAR_NEMOTRON_H_STATE_CONV)
                                        : 1u << PULSAR_NEMOTRON_H_STATE_KV);
    if (state_ok && s->mtp_written)
        state_ok = sp->slot[s->n_layer] == (1u << PULSAR_NEMOTRON_H_STATE_KV | 1u << PULSAR_NEMOTRON_H_STATE_MTP_PEND) &&
                   sp->drafter_at == pulsar_mtp_drafter_at && sp->drafter_frontier == pulsar_mtp_drafter_frontier;
    check(state_ok, "the declared state: ssm FIXED DELTA %llu B (+ %llu B a verify row), ssm_conv FIXED SNAPSHOT %llu B "
          "a Mamba-2 layer; kv POOL %llu B a token an attention layer%s", (unsigned long long)ssm.bytes,
          (unsigned long long)ssm.capture_bytes, (unsigned long long)conv.bytes, (unsigned long long)kv.bytes,
          s->mtp_written ? "; the drafter's slot: kv POOL + mtp_pend FIXED (the parked row), the lane's hooks" : "");
    const uint32_t want = PULSAR_FAMILY_CAP_CHAT | PULSAR_FAMILY_CAP_BANKS | PULSAR_FAMILY_CAP_SEGMENTS |
                          PULSAR_FAMILY_CAP_PAYLOAD | PULSAR_FAMILY_CAP_SPEC;
    check(e->family->caps == want && e->family->spec == &PULSAR_DRIVER_SPEC_TARGET &&
              e->family->driver->mtp && e->family->driver->absorb == pulsar_drafter_lane_absorb,
          "caps 0x%x (chat, banks, segments, payload, spec: the driver's verify target; no TP / images yet); the "
          "driver declares its MTP module, its lockstep the drafter lane's (MTP or block, L298)", e->family->caps);
    pulsar_engine_close(e);
    fprintf(stderr, "NEMOTRON-FAMILY PROBE %s\n", n_fail ? "FAIL" : "PASS");
    return n_fail ? 1 : 0;
}

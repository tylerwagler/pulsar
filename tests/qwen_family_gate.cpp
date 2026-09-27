/* L251 S1 qwen-family gate: the Qwen4-exp family skeleton, graded on what it
 * claims today -- it loads, plans, binds, sizes and allocates its session
 * state, and refuses every step by naming the first op not implemented yet.
 *
 * Inputs are zero-weight containers made by tests/qwen_family_container.py
 * from the HF checkpoint's own headers and config (the container contract,
 * family_qwen.h section 2): good.safetensors plus four mutants the loader must
 * refuse by name (arch, shape, tensor, layer-type).  Nothing reads a weight.
 *
 *   ./tests/qwen_family_gate DIR          host part (no GPU needed)
 *   ./tests/qwen_family_gate DIR --gpu    + a real session on the device
 *
 * The --gpu part opens the container for inspection and then initialises the
 * device on that engine by hand: the skeleton's session needs no weights on
 * the GPU (every op refuses before any runs), and staging 234 GiB of sparse
 * zeros would measure nothing. */
#include "pulsar_engine_internal.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* Present-but-never-called ops for the order probe: the walk only asks
 * whether an entry is set. */
static bool never_embed(const pulsar_qwen_step *) { abort(); }
static bool never_layer(const pulsar_qwen_step *, uint32_t) { abort(); }
static bool never_gr(const pulsar_qwen_step *, uint32_t, pulsar_qwen_gr_side) { abort(); }

static pulsar_engine *open_inspect(const char *path) {
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = path;
    opt.backend = PULSAR_BACKEND_CUDA;
    opt.inspect_only = true;
    pulsar_engine *e = NULL;
    if (pulsar_engine_open(&e, &opt) != 0) return NULL;
    return e;
}

static void host_part(const char *dir) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/good.safetensors", dir);
    fprintf(stderr, "qwen-family gate: load %s\n", path);
    pulsar_engine *e = open_inspect(path);
    check(e != NULL, "the zero-weight container opens (--inspect)");
    if (!e) return;
    const pulsar_qwen_shape *s = &g_qwen_shape;
    check(pulsar_engine_family(e) == PULSAR_FAMILY_ID_QWEN4_EXP, "family is qwen4_exp (%s)",
          pulsar_engine_family_name(e));
    check(pulsar_engine_chat_format(e) == PULSAR_CHAT_QWEN && !pulsar_engine_chat_v41(e),
          "chat format is Qwen's, not a DeepSeek template");
    check(!pulsar_engine_has_tokenizer(e), "no tokenizer yet: front ends refuse it (S5 adds CAP_CHAT)");
    check(pulsar_engine_logits_width(e) == 248320, "logits width %d == 248320", pulsar_engine_logits_width(e));
    check(!strcmp(pulsar_engine_model_name(e), "Qwen3.8-Flash-Next"), "model name '%s'",
          pulsar_engine_model_name(e));
    check(pulsar_engine_model_id(e) == 2, "disk-KV model id %d is not a DeepSeek profile's", pulsar_engine_model_id(e));
    const pulsar_layer_plan *p = &e->plan;
    bool pattern = p->n_layer == 48;
    for (uint32_t il = 0; pattern && il < p->n_layer; il++)
        pattern = p->kind[il] == ((il % 4 == 3) ? PULSAR_LAYER_QWEN_QSA : PULSAR_LAYER_QWEN_GDN);
    check(pattern, "plan: %u layers, %u GDN + %u QSA, every 4th layer QSA", p->n_layer,
          pulsar_layer_plan_count(p, PULSAR_LAYER_QWEN_GDN), pulsar_layer_plan_count(p, PULSAR_LAYER_QWEN_QSA));
    check(s->ple_layer == 1, "PLE at 0-based layer %u (HF ple_layer_ids [2] is 1-based)", s->ple_layer);
    const pulsar_qwen_weights *w = e->qwen_weights;
    bool bound = w && w->token_embd && w->output && w->mixer.hc_norm && !w->mixer.inject;
    for (uint32_t il = 0; bound && il < p->n_layer; il++) {
        const pulsar_qwen_layer_weights *L = &w->layer[il];
        const bool gdn = p->kind[il] == PULSAR_LAYER_QWEN_GDN;
        bound = L->gr_attn.inject && L->gr_mlp.inject && L->moe_gate_up && L->moe_down && L->sh_gate_scalar &&
                (gdn ? (L->gdn_in_qkv && L->gdn_out && L->gdn_conv && !L->attn_q)
                     : (L->attn_q && L->attn_o && L->idx_qk && !L->gdn_in_qkv)) &&
                ((il == s->ple_layer) == (L->ple_key != NULL));
    }
    check(bound, "every layer binds its kind's tensors and only those; PLE only at layer %u", s->ple_layer);
    check(w && w->ple_multipliers[0] % 2 == 1 && w->ple_head_vocab[0] > s->ngram_vocab_base &&
          w->ple_head_offset[1] == w->ple_head_offset[0] + w->ple_head_vocab[0],
          "PLE tables: odd multipliers, table 0 of %llu rows (> %llu), offsets contiguous",
          w ? (unsigned long long)w->ple_head_vocab[0] : 0ull, (unsigned long long)s->ngram_vocab_base);

    /* The op table is COMPLETE: S2's gdn and S3's qsa landed 2026-09-27, so the
     * driver has no op left to name.  (A step still refuses -- the synthetic
     * fixture's tensors are not the recipe's formats -- but no longer for a
     * missing op; the probes below keep the walk's order honest.) */
    uint32_t at = 0;
    const pulsar_qwen_op_id miss = pulsar_qwen_first_missing_op(&g_qwen_ops, p, s, &at);
    check(miss == PULSAR_QWEN_OP_COUNT, "the op table is complete: no missing op (got %s)", pulsar_qwen_op_name(miss));
    /* The walk's order is the forward's: with only embed present the next is
     * gr_read at layer 0; with every op but qsa present it is qsa at layer 3. */
    pulsar_qwen_ops probe;
    memset(&probe, 0, sizeof(probe));
    probe.embed = never_embed;
    check(pulsar_qwen_first_missing_op(&probe, p, s, &at) == PULSAR_QWEN_OP_GR_READ && at == 0,
          "embed present -> gr_read at layer 0");
    probe.gr_read = never_gr;
    probe.gdn = never_layer;
    probe.gr_write = never_gr;
    probe.moe = never_layer;
    check(pulsar_qwen_first_missing_op(&probe, p, s, &at) == PULSAR_QWEN_OP_PLE && at == 1,
          "the PLE op is reached at layer 1, before its gr_read");
    probe.ple = never_layer;
    check(pulsar_qwen_first_missing_op(&probe, p, s, &at) == PULSAR_QWEN_OP_QSA && at == 3,
          "then qsa at layer 3");
    probe.qsa = never_layer;
    check(pulsar_qwen_first_missing_op(&probe, p, s, &at) == PULSAR_QWEN_OP_HEAD, "then head");

    /* The session state at Tyler's KV target, priced by the allocator run dry. */
    struct { uint32_t banks, ctx; } cfgs[] = { {1, 4096}, {1, 1572864}, {1, 2097152}, {8, 262144} };
    const uint64_t per_tok = 12ull * pulsar_qwen_kv_row_bytes(s);
    for (size_t i = 0; i < sizeof(cfgs) / sizeof(cfgs[0]); i++) {
        uint64_t managed = 0;
        const uint64_t bytes = pulsar_qwen_state_price(s, p, cfgs[i].banks, cfgs[i].ctx, 4096, &managed);
        const uint64_t want_managed = (uint64_t)cfgs[i].banks *
            (cfgs[i].ctx * per_tok + 12ull * ((cfgs[i].ctx + s->idx_block - 1) / s->idx_block) *
                                     pulsar_qwen_index_row_bytes(s));
        check(bytes > 0 && managed == want_managed,
              "state at %u bank(s) x %u tokens: %.2f GiB (%.2f GiB demand-paged KV + index = the row sizes)",
              cfgs[i].banks, cfgs[i].ctx, (double)bytes / 1073741824.0, (double)managed / 1073741824.0);
    }
    pulsar_engine_close(e);

    static const char *const mutants[] = { "arch", "shape", "tensor", "layer-type", "s4-format", "ple-rows" };
    for (size_t i = 0; i < sizeof(mutants) / sizeof(mutants[0]); i++) {
        snprintf(path, sizeof(path), "%s/%s.safetensors", dir, mutants[i]);
        fprintf(stderr, "qwen-family gate: mutant '%s' (the loader must refuse it by name):\n", mutants[i]);
        pulsar_engine *m = open_inspect(path);
        check(m == NULL, "mutant '%s' refused", mutants[i]);
        if (m) pulsar_engine_close(m);
    }
}

static void gpu_part(const char *dir) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/good.safetensors", dir);
    fprintf(stderr, "qwen-family gate: a session on the device\n");
    pulsar_engine *e = open_inspect(path);
    if (!e) {
        check(false, "the container opens for the device part");
        return;
    }
    e->gpu_ready = pulsar_gpu_init() != 0;
    check(e->gpu_ready, "device initialised");
    if (!e->gpu_ready) {
        pulsar_engine_close(e);
        return;
    }
    e->prefill_chunk = 4096;
    const int ctx = 262144;
    const uint64_t price = pulsar_engine_session_cost_bytes_banked(e, ctx, (int)gpu_graph_bank_pool_n());
    pulsar_session *sess = NULL;
    const int rc = pulsar_session_create(&sess, e, ctx);
    check(rc == 0 && sess && sess->qwen, "session created (%d tokens, %u bank(s))", ctx, gpu_graph_bank_pool_n());
    if (rc != 0 || !sess) {
        pulsar_engine_close(e);
        return;
    }
    check(pulsar_session_resident_bytes(sess) == price,
          "allocated %llu bytes == the dry price %llu", (unsigned long long)pulsar_session_resident_bytes(sess),
          (unsigned long long)price);

    int toks[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    pulsar_tokens prompt = { toks, 8, 8 };
    char err[256] = "";
    /* The forward RUNS now: every op is implemented (S2's gdn, S3's qsa, S4's
     * six) and the fixture carries the recipe's tensor formats, so the 8-token
     * prefill commits, the bank's position advances, and eval emits logits.
     * The fixture's weights are ZEROS: this proves the wiring and the buffer
     * shapes run end to end, NOT the numbers (the reference gate grades those). */
    check(pulsar_session_sync(sess, &prompt, err, sizeof(err)) == 0, "prefill ran: %s", err);
    check(pulsar_session_pos(sess) == 8, "the prefill committed 8 positions (pos %d)", pulsar_session_pos(sess));
    err[0] = '\0';
    check(pulsar_session_eval(sess, 1, err, sizeof(err)) == 0, "eval ran: %s", err);
    pulsar_multiseq_req row = { 0u, 0, 1 };
    float *logits = (float *)xmalloc(248320u * sizeof(float));
    err[0] = '\0';
    check(pulsar_session_decode_multiseq(sess, &row, 1, logits, 248320, err, sizeof(err)) != 0,
          "batched decode refused: %s", err);
    err[0] = '\0';
    check(pulsar_session_decode_mixed(sess, &row, 1, logits, 248320, NULL, 0, err, sizeof(err)) != 0,
          "mixed step refused: %s", err);
    free(logits);
    check(pulsar_session_bank_fork(sess, 0, 1, toks, 8, 0) != 0, "bank fork refused (no BANKS cap)");
    check(pulsar_session_payload_bytes(sess) == 0, "payload refused (no PAYLOAD cap)");
    check(pulsar_session_spec_next_base(sess, 0.0f, 0, 1.0f, 0.0f, NULL) < 0, "speculation refused (no SPEC cap)");
    pulsar_session_invalidate(sess);
    pulsar_session_free(sess);
    pulsar_engine_close(e);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s DIR [--gpu]\n", argv[0]);
        return 2;
    }
    const bool gpu = argc > 2 && !strcmp(argv[2], "--gpu");
    host_part(argv[1]);
    if (gpu) gpu_part(argv[1]);
    if (n_fail) {
        fprintf(stderr, "qwen-family gate: FAIL (%d)\n", n_fail);
        return 1;
    }
    fprintf(stderr, "qwen-family gate: PASS (%s)\n", gpu ? "host + device" : "host");
    return 0;
}

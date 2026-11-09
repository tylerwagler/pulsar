/* nemotron_enc_family.cpp -- the Nemotron-3.5 encoders' descriptor (family.h): load, the engine facts, the encoder
 * entries (PULSAR_FAMILY_CAP_ENCODE), and the session entries, which refuse by name -- an encoder has no session (no
 * next token, no KV to keep): its one operation is pulsar_engine_encode. */
/** @file nemotron_enc_family.cpp
 *  @brief The Nemotron-3.5 encoders' descriptor: load, the engine facts, and session entries that refuse (an encoder
 *  has no session). */
#include "nemotron_enc.h"
#include "engine/exl3_trellis.h"
#include "lib/bpe_tokenizer.h"

namespace {

/// The architecture name every refusal and announce line is prefixed with.
const char *const ARCH = PULSAR_NEMOTRON_ENC_ARCH;
/// The one reason every session entry refuses with.
const char *const NO_SESSION = "an encoder has no session (it returns a vector or a score per input, not a next "
                               "token): pulsar_engine_encode is its operation";

/** pulsar_family::unload: free the run, the tokenizer, the widened score head and the family state (safe after a
 *  partial load).
 *  @param e  the engine; its family_state is freed and set to NULL */
void unload(pulsar_engine *e) {
    pulsar_nemotron_enc_model *w = (pulsar_nemotron_enc_model *)e->family_state;
    if (w) {
        pulsar_nemotron_enc_run_free(w);
        bpe_tokenizer_free(w->tok);
        free(w->score_f32);
    }
    free(w);
    e->family_state = NULL;
}

/** pulsar_family::load: refuse the decoder options, then read the shape and plan, scan the artifact, bind and price
 *  every tensor, and load the tokenizer beside the shards.
 *  @param e    the engine; family_state and plan are written (family_state even on failure, for unload)
 *  @param opt  the open options (model_path locates tokenizer.json)
 *  @return true = loaded; false = refused (said) */
bool load(pulsar_engine *e, const pulsar_engine_options *opt) {
    /* the options that name a decoder's feature refuse here, by name, rather than being ignored */
    if (opt->drafter && opt->drafter[0]) {
        fprintf(stderr, "pulsar: %s: --drafter %s: an encoder does not decode; refusing\n", ARCH, opt->drafter);
        return false;
    }
    if (opt->expert_overlay && opt->expert_overlay[0]) {
        fprintf(stderr, "pulsar: %s: --expert-overlay is a DeepSeek routed-expert tool; refusing\n", ARCH);
        return false;
    }
    if (opt->rope_yarn_factor != 0.0) {
        fprintf(stderr, "pulsar: %s: --rope-yarn: the container declares its rope (YaRN, pinned in the shape); "
                        "refusing\n", ARCH);
        return false;
    }
    if (opt->directional_steering_file && opt->directional_steering_file[0]) {
        fprintf(stderr, "pulsar: %s: directional steering is not implemented for this family; refusing\n", ARCH);
        return false;
    }
    pulsar_nemotron_enc_model *w = (pulsar_nemotron_enc_model *)xcalloc(1, sizeof(*w));
    e->family_state = w;   /* unload frees it, whatever this load's verdict */
    pulsar_load_refusals_reset();   /* the loader's one failure policy (family.cpp) */
    /* stage 1: the shape and the plan */
    if (!pulsar_nemotron_enc_read_shape(&e->model, &w->shape) || pulsar_load_refusals()) {
        fprintf(stderr, "pulsar: %s: %u refusal(s) in the config -- the model does not load\n", ARCH,
                pulsar_load_refusals());
        return false;
    }
    pulsar_nemotron_enc_plan(&w->shape, &e->plan);
    /* stage 2: the whole-artifact scans every family's load runs */
    weights_reject_unsupported_types(&e->model);
    weights_reject_bad_e8m0(&e->model);
    if (pulsar_load_refusals()) {
        fprintf(stderr, "pulsar: %s: %u refusal(s) in the artifact scans -- the model does not load\n", ARCH,
                pulsar_load_refusals());
        return false;
    }
    /* stage 3: every tensor by its role, admitted, priced */
    if (!pulsar_nemotron_enc_bind(&e->model, w) || pulsar_load_refusals()) {
        fprintf(stderr, "pulsar: %s: %u refusal(s) binding the artifact -- the model does not load\n", ARCH,
                pulsar_load_refusals());
        return false;
    }
    /* stage 4: the tokenizer beside the shards */
    if (!pulsar_nemotron_enc_load_tokenizer(w, opt->model_path)) return false;
    const pulsar_nemotron_enc_shape *s = &w->shape;
    fprintf(stderr, "pulsar: %s: %s, %s: %u bidirectional layers (%uq/%ukv/%u, ffn %u), YaRN rope, mean pool -> %s; "
                    "inputs up to %u tokens\n",
            ARCH, s->name, s->kind == PULSAR_ENCODER_EMBED ? "Embed" : "Rerank", s->n_layer, s->n_head, s->n_head_kv,
            s->head_dim, s->n_ff, s->kind == PULSAR_ENCODER_EMBED ? "L2-normalised embedding" : "score",
            s->rope_orig_ctx);
    return true;
}

/** pulsar_family::after_gpu: build the forward's device state.
 *  @param e  the engine
 *  @return pulsar_nemotron_enc_run_create's verdict */
bool after_gpu(pulsar_engine *e) { return pulsar_nemotron_enc_run_create(e); }

/** The engine's family state as the encoder's (no family check: only this family's hooks call it).
 *  @param e  an engine this family loaded
 *  @return its pulsar_nemotron_enc_model */
const pulsar_nemotron_enc_model *model_of(const pulsar_engine *e) {
    return (const pulsar_nemotron_enc_model *)e->family_state;
}

/* no logits row: an encoder heads nothing */
/** pulsar_family::logits_width.
 *  @return 0 */
uint32_t logits_width(const pulsar_engine *) { return 0; }
/** pulsar_family::model_name.
 *  @param e  the engine
 *  @return the Embed or Rerank checkpoint name, by the shape's kind (static) */
const char *model_name(const pulsar_engine *e) {
    return model_of(e)->shape.kind == PULSAR_ENCODER_EMBED ? "Nemotron-3.5-Embed-8B" : "Nemotron-3.5-Rerank-8B";
}
/** pulsar_family::served_model_id.
 *  @param e  the engine
 *  @return "nemotron-embed" or "nemotron-rerank", by the shape's kind (static) */
const char *served_model_id(const pulsar_engine *e) {
    return model_of(e)->shape.kind == PULSAR_ENCODER_EMBED ? "nemotron-embed" : "nemotron-rerank";
}
/** pulsar_family::chat_format.
 *  @return PULSAR_CHAT_NONE: the encoder renders its own inputs */
pulsar_chat_format chat_format(const pulsar_engine *) { return PULSAR_CHAT_NONE; }
/* Disk-KV compatibility id: 0 and 1 DeepSeek's profiles, 2 Qwen's, 3 Inkling's, 4 Nemotron-H's (an encoder keeps no
 * KV; the id still names the model) */
/** pulsar_family::model_id.
 *  @return 5 */
int model_id(const pulsar_engine *) { return 5; }
/* the llama-4 q scale is exactly 1 below the original context: the encoder's input cap */
/** pulsar_family::trained_context.
 *  @param e  the engine
 *  @return the shape's rope_orig_ctx (16384), in tokens: the input cap */
uint64_t trained_context(const pulsar_engine *e) { return model_of(e)->shape.rope_orig_ctx; }
/** pulsar_family::drafter.
 *  @return PULSAR_DRAFTER_NONE */
pulsar_drafter_kind drafter(pulsar_engine *) { return PULSAR_DRAFTER_NONE; }
/* no routed experts: the dense linears' EXL3 tier in EXL3's value space (family.h quant_bits), 0 when BF16 */
/** pulsar_family::quant_bits: the highest EXL3 rate over every linear (attention q / k / v / o and MLP
 *  gate / up / down; the score head is BF16 by the binder).
 *  @param e  the engine
 *  @return 20 + K in half bits, or 0 when every linear is BF16 */
int quant_bits(pulsar_engine *e) {
    const pulsar_nemotron_enc_model *w = model_of(e);
    int bits = 0;
    for (uint32_t il = 0; il < w->shape.n_layer; il++) {
        const pulsar_nemotron_enc_layer &L = w->layer[il];
        for (const pulsar_tensor *t : {L.attn_q, L.attn_k, L.attn_v, L.attn_o, L.ffn_gate, L.ffn_up, L.ffn_down}) {
            const int k2 = t ? exl3_type_k2(t->type) : 0;
            if (k2 && 20 + k2 > bits) bits = 20 + k2;
        }
    }
    return bits;
}

/* ---- the session: none -------------------------------------------------------------------------------------------- */

/** pulsar_family_session_ops::create: refuses (NO_SESSION).
 *  @return always 1 (refused) */
int session_create(pulsar_session *, uint32_t) {
    fprintf(stderr, "pulsar: %s: %s -- refusing the session\n", ARCH, NO_SESSION);
    return 1;
}
/* create refuses, so no session of this family exists to step; the step entries say why anyway */
/** The step entries' shared refusal.
 *  @param err     receives "<arch>: NO_SESSION" when non-NULL
 *  @param errlen  err's capacity in bytes
 *  @return always -1 */
int step_refused(char *err, size_t errlen) {
    if (err) snprintf(err, errlen, "%s: %s", ARCH, NO_SESSION);
    return -1;
}
/** pulsar_family_session_ops::eval_row: refuses (no session can exist).
 *  @param err     the refusal text
 *  @param errlen  err's capacity in bytes
 *  @return always -1 */
int eval_row(pulsar_session *, int, char *err, size_t errlen) { return step_refused(err, errlen); }
/** pulsar_family_session_ops::decode_multiseq: refuses (no session can exist).
 *  @param err     the refusal text
 *  @param errlen  err's capacity in bytes
 *  @return always -1 */
int decode_multiseq(pulsar_session *, const pulsar_multiseq_req *, uint32_t, float *, int, char *err, size_t errlen) {
    return step_refused(err, errlen);
}
/** pulsar_family_session_ops::decode_mixed: refuses (no session can exist).
 *  @param err     the refusal text
 *  @param errlen  err's capacity in bytes
 *  @return always -1 */
int decode_mixed(pulsar_session *, const pulsar_multiseq_req *, uint32_t, float *, int, uint32_t *, uint32_t, char *err,
                 size_t errlen) {
    return step_refused(err, errlen);
}

/// The session table: create refuses, so the step entries only ever say why.
const pulsar_family_session_ops k_session_ops = {
    /* .create          = */ session_create,
    /* .destroy         = */ NULL,   /* no session is ever built (create refuses) */
    /* .sync            = */ NULL,
    /* .eval_row        = */ eval_row,
    /* .decode_multiseq = */ decode_multiseq,
    /* .decode_mixed    = */ decode_mixed,
    /* .invalidate      = */ NULL,
    /* .decode_fused    = */ NULL,
    /* .fused_heads_max = */ 0,
};

/* The bank pool: every hook takes a session, and none can exist; no CAP_BANKS, so the engine-level price refuses by
 * name before it would ask. */
/// The bank table: every hook NULL (no session exists to bank; no CAP_BANKS).
const pulsar_family_bank_ops k_bank_ops = {};

}  // namespace

const pulsar_family PULSAR_FAMILY_NEMOTRON_ENC = {
    /* .id           = */ PULSAR_FAMILY_ID_NEMOTRON_ENC,
    /* .arch         = */ PULSAR_NEMOTRON_ENC_ARCH,
    /* .name         = */ "Nemotron-3.5 encoder",
    /* .caps         = */ PULSAR_FAMILY_CAP_ENCODE,   /* no CHAT: the encoder renders its own inputs (encode_render) */
    /* .load         = */ load,
    /* .after_gpu    = */ after_gpu,
    /* .logits_width = */ logits_width,
    /* .model_name   = */ model_name,
    /* .served_id    = */ served_model_id,
    /* .chat_format  = */ chat_format,
    /* .model_id     = */ model_id,
    /* .trained_ctx  = */ trained_context,
    /* .rope_id      = */ pulsar_family_rope_fixed,
    /* .tp_shape     = */ NULL,   /* no CAP_TP: the open refuses tensor parallelism before it would ask */
    /* .drafter      = */ drafter,
    /* .quant_bits   = */ quant_bits,
    /* .spec         = */ NULL,
    /* .session      = */ &k_session_ops,
    /* .tokenizer    = */ NULL,   /* no chat front: pulsar_engine_has_tokenizer is false, a chat server refuses it */
    /* .tp_slices    = */ NULL,
    /* .act_kind     = */ PULSAR_ACT_KIND_ROWS,
    /* .vision       = */ NULL,   /* v1: text; the Pixtral tower is not written */
    /* .banks        = */ &k_bank_ops,
    /* .imatrix      = */ NULL,
    /* .driver       = */ NULL,
    /* .unload       = */ unload,
    /* .encoder      = */ &PULSAR_NEMOTRON_ENC_ENCODER,
};

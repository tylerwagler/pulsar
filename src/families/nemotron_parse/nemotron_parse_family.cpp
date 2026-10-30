/* nemotron_parse_family.cpp -- Parse's descriptor (family.h): load, the engine facts, the parser entry
 * (PULSAR_FAMILY_CAP_PARSE), and the session entries, which refuse by name -- a page is one call, not a session. */
/** @file nemotron_parse_family.cpp
 *  @brief Parse's descriptor: load, the engine facts, the parser entry, and session entries that refuse (a page is one
 *  call). */
#include "nemotron_parse.h"

namespace {

/// The architecture name every refusal and announce line is prefixed with.
const char *const ARCH = PULSAR_NEMOTRON_PARSE_ARCH;
/// The one reason every session entry refuses with.
const char *const NO_SESSION = "Parse has no session (a page image in, its text out): pulsar_engine_parse_page is its "
                               "operation";

/** pulsar_family::unload: free the run, the decode table and the family state (safe after a partial load).
 *  @param e  the engine; its family_state is freed and set to NULL */
void unload(pulsar_engine *e) {
    pulsar_nemotron_parse_model *w = (pulsar_nemotron_parse_model *)e->family_state;
    if (w) {
        pulsar_nemotron_parse_run_free(w);
        delete w->vocab;
        delete w;
    }
    e->family_state = NULL;
}

/** pulsar_family::load: refuse the decoder-only options, read and pin the facts, plan the decoder's layers, scan the
 *  artifact, bind and price every tensor, and read the decode table beside the shards.
 *  @param e    the engine; family_state and plan are written (family_state even on failure, for unload)
 *  @param opt  the open options (model_path locates tokenizer.json)
 *  @return true = loaded; false = refused (said) */
bool load(pulsar_engine *e, const pulsar_engine_options *opt) {
    if (opt->drafter && opt->drafter[0]) {
        fprintf(stderr, "pulsar: %s: --drafter %s: Parse's one-step MTP head is not part of this declaration (v1); "
                        "refusing\n", ARCH, opt->drafter);
        return false;
    }
    if (opt->expert_overlay && opt->expert_overlay[0]) {
        fprintf(stderr, "pulsar: %s: --expert-overlay is a DeepSeek routed-expert tool; refusing\n", ARCH);
        return false;
    }
    if (opt->rope_yarn_factor != 0.0) {
        fprintf(stderr, "pulsar: %s: --rope-yarn: this family has no rope to scale; refusing\n", ARCH);
        return false;
    }
    if (opt->directional_steering_file && opt->directional_steering_file[0]) {
        fprintf(stderr, "pulsar: %s: directional steering is not implemented for this family; refusing\n", ARCH);
        return false;
    }
    pulsar_nemotron_parse_model *w = new pulsar_nemotron_parse_model();
    e->family_state = w;   /* unload frees it, whatever this load's verdict */
    pulsar_load_refusals_reset();
    if (!pulsar_nemotron_parse_read_shape(&e->model, &w->shape) || pulsar_load_refusals()) {
        fprintf(stderr, "pulsar: %s: %u refusal(s) in the facts -- the model does not load\n", ARCH,
                pulsar_load_refusals());
        return false;
    }
    /* the plan is the decoder's layers (the tower is the family's own front, run once a page) */
    memset(&e->plan, 0, sizeof(e->plan));
    e->plan.n_layer = w->shape.n_layer;
    for (uint32_t il = 0; il < w->shape.n_layer; il++) {
        e->plan.kind[il] = PULSAR_LAYER_BLOCKS;
        e->plan.desc[il].mixer = PULSAR_MIXER_ATTN_FULL;
        e->plan.desc[il].ffn = PULSAR_FFN_DENSE;
    }
    weights_reject_unsupported_types(&e->model);
    weights_reject_bad_e8m0(&e->model);
    if (pulsar_load_refusals()) {
        fprintf(stderr, "pulsar: %s: %u refusal(s) in the artifact scans -- the model does not load\n", ARCH,
                pulsar_load_refusals());
        return false;
    }
    if (!pulsar_nemotron_parse_bind(&e->model, w) || pulsar_load_refusals()) {
        fprintf(stderr, "pulsar: %s: %u refusal(s) binding the artifact -- the model does not load\n", ARCH,
                pulsar_load_refusals());
        return false;
    }
    if (!pulsar_nemotron_parse_load_vocab(w, opt->model_path)) return false;
    const pulsar_nemotron_parse_shape *s = &w->shape;
    fprintf(stderr, "pulsar: %s: Nemotron-3.5 Parse: RADIO ViT %u x %u (%u heads of %u), neck -> %u, "
                    "mBART decoder %u x %u (%uq/%ukv), causal visual prefix; pages up to %u x %u, greedy + repetition "
                    "penalty %.2f, up to %u new tokens\n",
            ARCH, s->vit_depth, s->vit_width, s->vit_heads, s->vit_head_dim, s->neck_width,
            s->n_layer, s->width, s->heads, s->kv_heads, s->canvas_w, s->canvas_h, (double)s->rep_penalty, s->max_new);
    return true;
}

/** pulsar_family::after_gpu: build the forward's device state.
 *  @param e  the engine
 *  @return pulsar_nemotron_parse_run_create's verdict */
bool after_gpu(pulsar_engine *e) { return pulsar_nemotron_parse_run_create(e); }

/** The engine's family state as Parse's (no family check: only this family's hooks call it).
 *  @param e  an engine this family loaded
 *  @return its pulsar_nemotron_parse_model */
const pulsar_nemotron_parse_model *model_of(const pulsar_engine *e) {
    return (const pulsar_nemotron_parse_model *)e->family_state;
}
/** pulsar_family::logits_width.
 *  @param e  the engine
 *  @return the decoder's vocab (72512); the logits stay inside parse, no session reads them */
uint32_t logits_width(const pulsar_engine *e) { return model_of(e)->shape.vocab; }
/** pulsar_family::model_name.
 *  @return "Nemotron-3.5-Parse" (static) */
const char *model_name(const pulsar_engine *) { return "Nemotron-3.5-Parse"; }
/** pulsar_family::served_model_id.
 *  @return "nemotron-parse" (static) */
const char *served_model_id(const pulsar_engine *) { return "nemotron-parse"; }
/** pulsar_family::chat_format.
 *  @return PULSAR_CHAT_NONE */
pulsar_chat_format chat_format(const pulsar_engine *) { return PULSAR_CHAT_NONE; }
/* Disk-KV compatibility id: 0 and 1 DeepSeek's, 2 Qwen's, 3 Inkling's, 4 Nemotron-H's, 5 the encoders' */
/** pulsar_family::model_id.
 *  @return 6 */
int model_id(const pulsar_engine *) { return 6; }
/** pulsar_family::trained_context.
 *  @param e  the engine
 *  @return the decoder's max_seq (16384), in tokens */
uint64_t trained_context(const pulsar_engine *e) { return model_of(e)->shape.max_seq; }
/** pulsar_family::drafter.
 *  @return PULSAR_DRAFTER_NONE */
pulsar_drafter_kind drafter(pulsar_engine *) { return PULSAR_DRAFTER_NONE; }
/** pulsar_family::quant_bits.
 *  @return 0: no routed experts */
int quant_bits(pulsar_engine *) { return 0; }   /* no routed experts */

/** pulsar_family_session_ops::create: refuses (NO_SESSION).
 *  @return always 1 (refused) */
int session_create(pulsar_session *, uint32_t) {
    fprintf(stderr, "pulsar: %s: %s -- refusing the session\n", ARCH, NO_SESSION);
    return 1;
}
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
    /* .destroy         = */ NULL,
    /* .sync            = */ NULL,
    /* .eval_row        = */ eval_row,
    /* .decode_multiseq = */ decode_multiseq,
    /* .decode_mixed    = */ decode_mixed,
    /* .invalidate      = */ NULL,
    /* .decode_fused    = */ NULL,
    /* .fused_heads_max = */ 0,
};

/// The bank table: every hook NULL (no session exists to bank).
const pulsar_family_bank_ops k_bank_ops = {};

}  // namespace

const pulsar_family PULSAR_FAMILY_NEMOTRON_PARSE = {
    /* .id           = */ PULSAR_FAMILY_ID_NEMOTRON_PARSE,
    /* .arch         = */ PULSAR_NEMOTRON_PARSE_ARCH,
    /* .name         = */ "Nemotron-3.5 Parse",
    /* .caps         = */ PULSAR_FAMILY_CAP_PARSE,
    /* .load         = */ load,
    /* .after_gpu    = */ after_gpu,
    /* .logits_width = */ logits_width,
    /* .model_name   = */ model_name,
    /* .served_id    = */ served_model_id,
    /* .chat_format  = */ chat_format,
    /* .model_id     = */ model_id,
    /* .trained_ctx  = */ trained_context,
    /* .rope_id      = */ pulsar_family_rope_fixed,
    /* .tp_shape     = */ NULL,
    /* .drafter      = */ drafter,
    /* .quant_bits   = */ quant_bits,
    /* .spec         = */ NULL,
    /* .session      = */ &k_session_ops,
    /* .tokenizer    = */ NULL,   /* its decode table is the parser's own (nemotron_parse_weights.cpp) */
    /* .tp_slices    = */ NULL,
    /* .act_kind     = */ PULSAR_ACT_KIND_ROWS,
    /* .vision       = */ NULL,   /* the page front is the parser's, not the chat image path */
    /* .banks        = */ &k_bank_ops,
    /* .imatrix      = */ NULL,
    /* .driver       = */ NULL,
    /* .unload       = */ unload,
    /* .encoder      = */ NULL,
    /* .parser       = */ &PULSAR_NEMOTRON_PARSE_PARSER,
};

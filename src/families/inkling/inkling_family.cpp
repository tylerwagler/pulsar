/* inkling_family.cpp -- the Inkling family's descriptor (family.h): load, the engine facts, and the session
 * entries, which refuse by name until the forward is wired (inkling.h). */
/** @file inkling_family.cpp
 *  @brief The Inkling family descriptor: load, the engine facts, and session entries that refuse until the forward is
 *  wired. */
#include "inkling.h"
#include "engine/exl3_trellis.h"
#include "lib/bpe_tokenizer.h"

namespace {

/// The architecture name every refusal and announce line is prefixed with.
const char *const ARCH = PULSAR_INKLING_ARCH;

/* What a session needs that is not built yet, named once for every entry that refuses for it.  The core pieces are
 * in (F4 the row-step driver, F3 the state kinds, F1 attention, F2 router, E3 SConv, E5 head); Inkling's use of them is
 * not: its pulsar_driver_ops (block[PULSAR_LAYER_BLOCKS] composing desc[il] from those blocks, embed, head) and its
 * state declaration.  pulsar_family::driver stays NULL until then. */
/// The one reason every session entry refuses with, until Inkling's driver blocks exist.
const char *const NOT_WIRED =
    "forward not yet wired: Inkling's driver blocks (pulsar_family::driver -- block[PULSAR_LAYER_BLOCKS] over F1 "
    "attention, F2 router, E3 SConv, E5 head on the core row-step driver) and its state declaration (F3 kinds)";

/** pulsar_family::unload: free the tokenizer and the family state load allocated (safe after a partial load).
 *  @param e  the engine; its family_state is freed and set to NULL */
void unload(pulsar_engine *e) {
    if (e->family_state) bpe_tokenizer_free(((pulsar_inkling_model *)e->family_state)->tok);
    free(e->family_state);
    e->family_state = NULL;
}

/** pulsar_family::load: refuse other families' options, then read the shape and plan, scan the artifact, bind and
 *  price every tensor, and load the tokenizer beside the shards.
 *  @param e    the engine; family_state and plan are written (family_state even on failure, for unload)
 *  @param opt  the open options (model_path locates the tokenizer files)
 *  @return true = loaded; false = refused (said, and counted through pulsar_load_refuse) */
bool load(pulsar_engine *e, const pulsar_engine_options *opt) {
    /* the options that name another family's feature refuse here, by name, rather than being ignored */
    if (opt->drafter && opt->drafter[0]) {
        fprintf(stderr, "pulsar: %s: --drafter %s: this family's container holds no drafter to choose; refusing\n", ARCH,
                opt->drafter);
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
    pulsar_inkling_model *w = (pulsar_inkling_model *)xcalloc(1, sizeof(*w));
    e->family_state = w;   /* unload frees it, whatever this load's verdict */
    pulsar_load_refusals_reset();   /* the loader's one failure policy (family.cpp) */
    /* stage 1: the shape and the plan */
    if (!pulsar_inkling_read_shape(&e->model, &w->shape) || pulsar_load_refusals()) {
        fprintf(stderr, "pulsar: %s: %u refusal(s) in the config -- the model does not load\n", ARCH,
                pulsar_load_refusals());
        return false;
    }
    pulsar_inkling_plan(&w->shape, &e->plan);
    /* stage 2: the whole-artifact scans every family's load runs (a type no reader takes, a NaN E8M0 scale, a
     * non-finite EXL3 scale) */
    weights_reject_unsupported_types(&e->model);
    weights_reject_bad_e8m0(&e->model);
    if (pulsar_load_refusals()) {
        fprintf(stderr, "pulsar: %s: %u refusal(s) in the artifact scans -- the model does not load\n", ARCH,
                pulsar_load_refusals());
        return false;
    }
    /* stage 3: every tensor by its role, admitted, priced */
    if (!pulsar_inkling_bind(&e->model, w) || pulsar_load_refusals()) {
        fprintf(stderr, "pulsar: %s: %u refusal(s) binding the artifact -- the model does not load\n", ARCH,
                pulsar_load_refusals());
        return false;
    }
    /* stage 4: the tokenizer beside the shards */
    if (!pulsar_inkling_load_tokenizer(w, opt->model_path)) return false;
    const pulsar_inkling_shape *s = &w->shape;
    uint32_t n_window = 0, n_dense = 0;
    for (uint32_t il = 0; il < e->plan.n_layer; il++) {
        n_window += e->plan.desc[il].mixer == PULSAR_MIXER_ATTN_WINDOW;
        n_dense += e->plan.desc[il].ffn == PULSAR_FFN_DENSE;
    }
    fprintf(stderr, "pulsar: %s: %s, %u layers: %u window (%u keys) + %u full attention, %u dense + %u MoE (%u of %u "
                    "experts + %u shared); drafter %s; tower %s\n",
            ARCH, s->name, s->n_layer, n_window, s->window, s->n_layer - n_window, n_dense, s->n_layer - n_dense,
            s->n_expert_used, s->n_expert, s->n_shared,
            s->mtp_written ? "bound (MTP layers; drafting is F6's)" : "not in the container",
            s->vision_written ? "bound (images are refused until E9's image front)" : "not in the container");
    return true;
}

/** pulsar_family::after_gpu: nothing to register on the device; announces that every session refuses.
 *  @return always true */
bool after_gpu(pulsar_engine *) {
    fprintf(stderr, "pulsar: %s: %s -- every session refuses\n", ARCH, NOT_WIRED);
    return true;
}

/** The engine's family state as Inkling's (no family check: only this family's hooks call it).
 *  @param e  an engine this family loaded
 *  @return its pulsar_inkling_model */
const pulsar_inkling_model *model_of(const pulsar_engine *e) { return (const pulsar_inkling_model *)e->family_state; }

/** pulsar_family::logits_width.
 *  @param e  the engine
 *  @return n_vocab (201024): the padded head width, not the tokenizer's table */
uint32_t logits_width(const pulsar_engine *e) { return model_of(e)->shape.n_vocab; }
/** pulsar_family::model_name.
 *  @param e  the engine
 *  @return the shape's profile name (static) */
const char *model_name(const pulsar_engine *e) { return model_of(e)->shape.name; }
/** pulsar_family::served_model_id.
 *  @return "inkling-small" (static) */
const char *served_model_id(const pulsar_engine *) { return "inkling-small"; }
/* F5's renderer (src/lib/inkling_chat.h): the engine's chat front (inkling_tokenizer.cpp) and the server's family
 * table (src/server/family_inkling.cpp) */
/** pulsar_family::chat_format.
 *  @return PULSAR_CHAT_INKLING */
pulsar_chat_format chat_format(const pulsar_engine *) { return PULSAR_CHAT_INKLING; }
/* Disk-KV compatibility id: 0 and 1 are DeepSeek's profiles, 2 Qwen's. */
/** pulsar_family::model_id.
 *  @return 3, the disk-KV compatibility id */
int model_id(const pulsar_engine *) { return 3; }
/* positions: no RoPE; the log-scaled global attention is trained to model_max_length */
/** pulsar_family::trained_context.
 *  @param e  the engine
 *  @return the shape's max_context, in tokens */
uint64_t trained_context(const pulsar_engine *e) { return model_of(e)->shape.max_context; }
/* the MTP layers are bound when written, but no drafter runs on them until F6 (the chained-MTP drafter) */
/** pulsar_family::drafter.
 *  @return PULSAR_DRAFTER_NONE */
pulsar_drafter_kind drafter(pulsar_engine *) { return PULSAR_DRAFTER_NONE; }
/* the routed experts' tier in EXL3's value space (family.h quant_bits): 20 + the highest rate in half bits */
/** pulsar_family::quant_bits: the highest EXL3 rate over the routed expert stacks.
 *  @param e  the engine
 *  @return 20 + K in half bits for EXL3 experts; 0 when no stack is EXL3 (BF16 experts, or no MoE layer) */
int quant_bits(pulsar_engine *e) {
    const pulsar_inkling_model *w = model_of(e);
    int bits = 0;
    for (uint32_t il = 0; il < w->shape.n_layer; il++)
        for (const pulsar_tensor *t : {w->layer[il].exp_gate, w->layer[il].exp_up, w->layer[il].exp_down}) {
            const int k2 = t ? exl3_type_k2(t->type) : 0;
            if (k2 && 20 + k2 > bits) bits = 20 + k2;
        }
    return bits;
}

/* ---- the session: refused by name until the driver blocks -------------------------------------------------------------------- */

/** pulsar_family_session_ops::create: refuses (NOT_WIRED).
 *  @return always 1 (refused) */
int session_create(pulsar_session *, uint32_t) {
    fprintf(stderr, "pulsar: %s: %s -- refusing the session\n", ARCH, NOT_WIRED);
    return 1;
}
/* create refuses, so no session of this family exists to destroy, sync or step; the step entries say why anyway */
/** The step entries' shared refusal.
 *  @param err     receives "<arch>: NOT_WIRED" when non-NULL
 *  @param errlen  err's capacity in bytes
 *  @return always -1 */
int step_refused(char *err, size_t errlen) {
    if (err) snprintf(err, errlen, "%s: %s", ARCH, NOT_WIRED);
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
    /* .sync            = */ NULL,   /* F3 / F4: the state kinds and the driver's prefill chunk */
    /* .eval_row        = */ eval_row,
    /* .decode_multiseq = */ decode_multiseq,
    /* .decode_mixed    = */ decode_mixed,
    /* .invalidate      = */ NULL,
    /* .decode_fused    = */ NULL,
    /* .fused_heads_max = */ 0,
};

/* The bank pool: every hook takes a session, and none can exist (create refuses); the family declares no CAP_BANKS,
 * so the engine-level price (pulsar_engine_demand_paged_bytes_per_bank) refuses by name before it would ask.  F3's
 * declared state kinds give the core what it derives these from. */
/// The bank table: every hook NULL (no session exists to bank; no CAP_BANKS).
const pulsar_family_bank_ops k_bank_ops = {};

}  // namespace

const pulsar_family PULSAR_FAMILY_INKLING = {
    /* .id           = */ PULSAR_FAMILY_ID_INKLING,
    /* .arch         = */ PULSAR_INKLING_ARCH,
    /* .name         = */ "Inkling",
    /* .caps         = */ PULSAR_FAMILY_CAP_CHAT,   /* the tokenizer + chat front (E6); the forward (F4), SPEC (F6), TP
                                                     * (E8), VISION (E9) next */
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
    /* .tokenizer    = */ &PULSAR_INKLING_TOKENIZER,   /* inkling_tokenizer.cpp (F5's BPE engine + renderer) */
    /* .tp_slices    = */ NULL,
    /* .act_kind     = */ PULSAR_ACT_KIND_ROWS,
    /* .vision       = */ NULL,   /* E9: the hmlp image front; the tower binds today, images are refused */
    /* .banks        = */ &k_bank_ops,
    /* .imatrix      = */ NULL,
    /* .driver       = */ NULL,   /* its blocks on the core driver are not written yet (NOT_WIRED) */
    /* .unload       = */ unload,
};

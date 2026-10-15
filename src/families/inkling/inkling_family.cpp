/* inkling_family.cpp -- the Inkling family's descriptor (family.h): load, the engine facts, and the session
 * entries, which refuse by name until the forward is wired (inkling.h). */
#include "inkling.h"
#include "engine/exl3_trellis.h"
#include "lib/bpe_tokenizer.h"

namespace {

const char *const ARCH = PULSAR_INKLING_ARCH;

/* What a session needs that is not built yet, named once for every entry that refuses for it.  The core pieces are
 * in (F4 the row-step driver, F3 the state kinds, F1 attention, F2 router, E3 SConv, E5 head); Inkling's use of them is
 * not: its pulsar_driver_ops (block[PULSAR_LAYER_BLOCKS] composing desc[il] from those blocks, embed, head) and its
 * state declaration.  pulsar_family::driver stays NULL until then. */
const char *const NOT_WIRED =
    "forward not yet wired: Inkling's driver blocks (pulsar_family::driver -- block[PULSAR_LAYER_BLOCKS] over F1 "
    "attention, F2 router, E3 SConv, E5 head on the core row-step driver) and its state declaration (F3 kinds)";

void unload(pulsar_engine *e) {
    if (e->family_state) bpe_tokenizer_free(((pulsar_inkling_model *)e->family_state)->tok);
    free(e->family_state);
    e->family_state = NULL;
}

bool load(pulsar_engine *e, const pulsar_engine_options *opt) {
    /* the options that name another family's feature refuse here, by name, rather than being ignored */
    if (opt->expert_overlay && opt->expert_overlay[0]) {
        fprintf(stderr, "pulsar: %s: --expert-overlay is a DeepSeek routed-expert tool; refusing\n", ARCH);
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

bool after_gpu(pulsar_engine *) {
    fprintf(stderr, "pulsar: %s: %s -- every session refuses\n", ARCH, NOT_WIRED);
    return true;
}

const pulsar_inkling_model *model_of(const pulsar_engine *e) { return (const pulsar_inkling_model *)e->family_state; }

uint32_t logits_width(const pulsar_engine *e) { return model_of(e)->shape.n_vocab; }
const char *model_name(const pulsar_engine *e) { return model_of(e)->shape.name; }
const char *served_model_id(const pulsar_engine *) { return "inkling-small"; }
/* F5's renderer (src/lib/inkling_chat.h): the engine's chat front (inkling_tokenizer.cpp) and the server's family
 * table (src/server/family_inkling.cpp) */
pulsar_chat_format chat_format(const pulsar_engine *) { return PULSAR_CHAT_INKLING; }
/* Disk-KV compatibility id: 0 and 1 are DeepSeek's profiles, 2 Qwen's. */
int model_id(const pulsar_engine *) { return 3; }
/* positions: no RoPE; the log-scaled global attention is trained to model_max_length */
uint64_t trained_context(const pulsar_engine *e) { return model_of(e)->shape.max_context; }
/* the MTP layers are bound when written, but no drafter runs on them until F6 (the chained-MTP drafter) */
pulsar_drafter_kind drafter(pulsar_engine *) { return PULSAR_DRAFTER_NONE; }
/* the routed experts' tier in EXL3's value space (family.h quant_bits): 20 + the highest rate in half bits */
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

int session_create(pulsar_session *, uint32_t) {
    fprintf(stderr, "pulsar: %s: %s -- refusing the session\n", ARCH, NOT_WIRED);
    return 1;
}
/* create refuses, so no session of this family exists to destroy, sync or step; the step entries say why anyway */
int step_refused(char *err, size_t errlen) {
    if (err) snprintf(err, errlen, "%s: %s", ARCH, NOT_WIRED);
    return -1;
}
int eval_row(pulsar_session *, int, char *err, size_t errlen) { return step_refused(err, errlen); }
int decode_multiseq(pulsar_session *, const pulsar_multiseq_req *, uint32_t, float *, int, char *err, size_t errlen) {
    return step_refused(err, errlen);
}
int decode_mixed(pulsar_session *, const pulsar_multiseq_req *, uint32_t, float *, int, uint32_t *, uint32_t, char *err,
                 size_t errlen) {
    return step_refused(err, errlen);
}

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

/* inkling_family.cpp -- the Inkling family's descriptor (family.h): load, the engine facts, and the session -- the
 * core row-step driver over Inkling's blocks (inkling_forward.cpp) and its declared state (state_kinds.h). */
#include "inkling.h"
#include "engine/exl3_trellis.h"
#include "lib/bpe_tokenizer.h"

namespace {

const char *const ARCH = PULSAR_INKLING_ARCH;

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
    /* the session's declared state (F3) and the digest its disk segments are keyed by: the directory's names, types
     * and sizes */
    pulsar_inkling_state_plan(e);
    {
        uint64_t h = 1469598103934665603ull;
        auto mix = [&h](const void *p, size_t n) {
            for (size_t i = 0; i < n; i++) { h ^= ((const uint8_t *)p)[i]; h *= 1099511628211ull; }
        };
        for (uint64_t i = 0; i < e->model.n_tensors; i++) {
            const pulsar_tensor *t = &e->model.tensors[i];
            mix(t->name.ptr, t->name.len);
            mix(&t->type, sizeof(t->type));
            mix(&t->bytes, sizeof(t->bytes));
        }
        w->artifact_digest = h;
    }
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
    fprintf(stderr, "pulsar: %s: no chat front yet (E6: the turn-by-turn render, the server table) -- the front ends "
                    "refuse this engine; the tokenizer serves ids and text\n", ARCH);
    return true;
}

/* The lane, announced once at open (rule 5): the core driver over Inkling's blocks, and what its table lacks for the
 * plan (a session refuses with exactly that). */
bool after_gpu(pulsar_engine *e) {
    uint32_t at = 0;
    const char *miss = pulsar_driver_first_missing(e->family->driver, &e->plan, &at);
    fprintf(stderr, "pulsar: %s: the core step driver over %u layers of shared blocks (F1 GQA, E3 SConv, F2 router + "
                    "rows MoE, E5 head); state: sconv FIXED + kv_win RING + kv POOL%s%s\n", ARCH,
            pulsar_layer_plan_count(&e->plan, PULSAR_LAYER_BLOCKS), miss ? "; MISSING " : "", miss ? miss : "");
    return true;
}

const pulsar_inkling_model *model_of(const pulsar_engine *e) { return (const pulsar_inkling_model *)e->family_state; }

uint32_t logits_width(const pulsar_engine *e) { return model_of(e)->shape.n_vocab; }
const char *model_name(const pulsar_engine *e) { return model_of(e)->shape.name; }
const char *served_model_id(const pulsar_engine *) { return "inkling-small"; }
/* No chat format exists for Inkling until E6 wires F5's renderer (src/lib/inkling_chat.h) and a server table (a
 * PULSAR_CHAT_INKLING id, its case in chat_family.cpp's family switch): the family declares no CAP_CHAT
 * and no tokenizer, so a front end refuses the engine before it asks (pulsar_engine_has_tokenizer).  One that asks
 * anyway is a caller bug, not an artifact's: no other family's template is a stand-in (rule 9). */
pulsar_chat_format chat_format(const pulsar_engine *) {
    pulsar_die("inkling: the family has no chat format until E6 wires F5's renderer; a front end asked past "
               "pulsar_engine_has_tokenizer");
    return PULSAR_CHAT_DS4_V4;   /* unreachable: pulsar_die exits */
}
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

/* ---- the session: the core row-step driver (step_driver.cpp) over Inkling's ops (inkling_forward.cpp) ----------- */

const pulsar_sync_ops k_sync = PULSAR_DRIVER_SYNC_OPS(PULSAR_INKLING_ARCH);
const pulsar_family_session_ops k_session_ops = PULSAR_DRIVER_SESSION_OPS(&k_sync);

}  // namespace

const pulsar_family PULSAR_FAMILY_INKLING = {
    /* .id           = */ PULSAR_FAMILY_ID_INKLING,
    /* .arch         = */ PULSAR_INKLING_ARCH,
    /* .name         = */ "Inkling",
    /* .caps         = */ PULSAR_FAMILY_CAP_BANKS | PULSAR_FAMILY_CAP_SEGMENTS | PULSAR_FAMILY_CAP_PAYLOAD,
                          /* the driver's session ops over the declared state; then CHAT (E6), SPEC (F6), TP (E8),
                           * VISION (E9) */
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
    /* .tokenizer    = */ &PULSAR_INKLING_TOKENIZER,   /* inkling_tokenizer.cpp (F5's BPE engine); no CAP_CHAT yet */
    /* .tp_slices    = */ NULL,
    /* .act_kind     = */ PULSAR_ACT_KIND_ROWS,
    /* .vision       = */ NULL,   /* E9: the hmlp image front; the tower binds today, images are refused */
    /* .banks        = */ &PULSAR_BANK_OPS_DECLARED,   /* state_kinds.cpp over the declared state */
    /* .imatrix      = */ NULL,
    /* .driver       = */ &PULSAR_INKLING_DRIVER,   /* inkling_forward.cpp */
    /* .unload       = */ unload,
};

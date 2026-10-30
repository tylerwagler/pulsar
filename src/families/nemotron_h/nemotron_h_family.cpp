/* nemotron_h_family.cpp -- the Nemotron-H family's descriptor (family.h): load, the engine facts, and the session --
 * the core row-step driver over the family's blocks (nemotron_h_forward.cpp) and its declared state (state_kinds.h). */
/** @file nemotron_h_family.cpp
 *  @brief The Nemotron-H family descriptor: load (drafter selection included), the engine facts and the driver's
 *  session. */
#include "nemotron_h.h"
#include "engine/exl3_trellis.h"
#include "lib/bpe_tokenizer.h"

namespace {

/// The architecture name every refusal and announce line is prefixed with.
const char *const ARCH = PULSAR_NEMOTRON_H_ARCH;

/** pulsar_family::unload: free the tokenizer, the block drafter's record and the family state, and clear the engine's
 *  block-drafter pointer (safe after a partial load).
 *  @param e  the engine; its family_state is freed and set to NULL */
void unload(pulsar_engine *e) {
    if (e->family_state) {
        bpe_tokenizer_free(((pulsar_nemotron_h_model *)e->family_state)->tok);
        free(((pulsar_nemotron_h_model *)e->family_state)->block);
    }
    e->block_drafter = NULL;
    free(e->family_state);
    e->family_state = NULL;
}

/** pulsar_family::load: refuse other families' options, read the shape and plan, select the drafter (MTP, LiLiCorr or
 *  none) from those the container holds and --drafter, scan the artifact, bind and price every tensor, load the
 *  tokenizer, then declare the session's state and digest the artifact for its disk segments.
 *  @param e    the engine; family_state, plan, state_plan, drafter_ops and block_drafter are written
 *  @param opt  the open options (drafter, dspark_disable, model_path)
 *  @return true = loaded; false = refused (said) */
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
    if (opt->rope_yarn_factor != 0.0) {
        fprintf(stderr, "pulsar: %s: --rope-yarn: this family's trunk has no rope to scale; refusing\n", ARCH);
        return false;
    }
    pulsar_nemotron_h_model *w = (pulsar_nemotron_h_model *)xcalloc(1, sizeof(*w));
    e->family_state = w;   /* unload frees it, whatever this load's verdict */
    pulsar_load_refusals_reset();   /* the loader's one failure policy (family.cpp) */
    /* stage 1: the shape and the plan */
    if (!pulsar_nemotron_h_read_shape(&e->model, &w->shape) || pulsar_load_refusals()) {
        fprintf(stderr, "pulsar: %s: %u refusal(s) in the config -- the model does not load\n", ARCH,
                pulsar_load_refusals());
        return false;
    }
    pulsar_nemotron_h_plan(&w->shape, &e->plan);
    /* L298: the drafter the open selects among those the container holds -- and so the one that is LOADED: the other
     * is declared unloaded by the bind (never staged; its memory stays the KV pool's) */
    {
        const pulsar_nemotron_h_shape &s = w->shape;
        const char *want = opt->drafter;
        bool use_mtp = false, use_block = false;
        const char *why = NULL;
        if (opt->dspark_disable) {
        } else if (!want || !want[0]) {
            if (s.mtp_written && s.block_written) why = "the container holds both the MTP module and LiLiCorr: pass --drafter mtp|lilicorr";
            use_mtp = s.mtp_written;
            use_block = s.block_written;
        } else if (!strcmp(want, "lilicorr")) {
            if (!s.block_written) why = "--drafter lilicorr: the container holds no LiLiCorr drafter (pulsar.drafter)";
            use_block = true;
        } else if (!strcmp(want, "mtp")) {
            if (!s.mtp_written) why = "--drafter mtp: the container holds no MTP module (pulsar.mtp_present)";
            use_mtp = true;
        } else {
            why = "--drafter names no drafter this family runs (mtp, lilicorr)";
        }
        if (why) {
            fprintf(stderr, "pulsar: %s: %s -- refusing\n", ARCH, why);
            return false;
        }
        w->drafting = use_mtp;
        if (use_block) {
            w->block = (pulsar_block_drafter *)xcalloc(1, sizeof(*w->block));
            pulsar_block_drafter *d = w->block;
            if (!pulsar_block_drafter_read(&e->model, d)) return false;
            /* the taps: the residual after target block tap_block[j], at its place in the plan */
            for (uint32_t j = 0; j < d->n_taps; j++) {
                if (d->tap_block[j] >= s.n_blocks) {
                    fprintf(stderr, "pulsar: %s: the drafter taps block %u of a %u-block target -- refusing\n", ARCH,
                            d->tap_block[j], s.n_blocks);
                    return false;
                }
                d->tap_layer[j] = s.block_layer[d->tap_block[j]];
                d->tap_sub[j] = s.block_sub[d->tap_block[j]];
            }
            if (d->hidden != s.n_embd || d->vocab != s.n_vocab) {
                fprintf(stderr, "pulsar: %s: the drafter's hidden %u / vocabulary %u are not the target's %u / %u -- "
                                "refusing\n", ARCH, d->hidden, d->vocab, s.n_embd, s.n_vocab);
                return false;
            }
        }
    }
    /* stage 2: the whole-artifact scans every family's load runs */
    weights_reject_unsupported_types(&e->model);
    weights_reject_bad_e8m0(&e->model);
    if (pulsar_load_refusals()) {
        fprintf(stderr, "pulsar: %s: %u refusal(s) in the artifact scans -- the model does not load\n", ARCH,
                pulsar_load_refusals());
        return false;
    }
    /* stage 3: every tensor by its role, admitted, priced */
    if (!pulsar_nemotron_h_bind(&e->model, w) || pulsar_load_refusals()) {
        fprintf(stderr, "pulsar: %s: %u refusal(s) binding the artifact -- the model does not load\n", ARCH,
                pulsar_load_refusals());
        return false;
    }
    /* stage 4: the tokenizer beside the shards */
    if (!pulsar_nemotron_h_load_tokenizer(w, opt->model_path)) return false;
    /* N9 / L298: the selected drafter behind the round API (none when the open turned drafting off) */
    if (w->block) w->block->embed = w->embed;
    e->block_drafter = w->block;
    e->drafter_ops = w->drafting ? &PULSAR_NEMOTRON_H_DRAFTER : w->block ? &w->block->ops : NULL;
    /* the session's declared state (N3, F3) and the digest its disk segments are keyed by: the directory's names,
     * types and sizes */
    pulsar_nemotron_h_state_plan(e);
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
    const pulsar_nemotron_h_shape *s = &w->shape;
    uint32_t n_mamba = 0, n_moe = 0;
    for (uint32_t il = 0; il < e->plan.n_layer; il++) {
        n_mamba += e->plan.desc[il].mixer == PULSAR_MIXER_MAMBA2;
        n_moe += e->plan.desc[il].ffn == PULSAR_FFN_MOE;
    }
    fprintf(stderr, "pulsar: %s: %s, %u blocks -> %u layers: %u Mamba-2 + %u attention mixers, %u MoE (%u of %u experts "
                    "on a %u latent + 1 shared) + %u mixer-only; drafter %s; vision not in the container (v1)\n",
            ARCH, s->name, s->n_blocks, s->n_layer, n_mamba, s->n_layer - n_mamba, n_moe, s->n_expert_used, s->n_expert,
            s->latent, s->n_layer - n_moe,
            w->drafting ? "the MTP module (the chained-MTP drafter)"
            : w->block  ? "LiLiCorr (the block drafter; the MTP module unloaded)"
            : s->mtp_written || s->block_written ? "in the container, unloaded (drafting off)"
                                                 : "not in the container");
    if (w->block) {
        char d[256];
        pulsar_block_drafter_describe(w->block, d, sizeof d);
        fprintf(stderr, "pulsar: %s: %s\n", ARCH, d);
    }
    return true;
}

/* The lane, announced once at open (rule 5): the core driver over the family's blocks, and what its table lacks for
 * the plan (a session refuses with exactly that). */
/** @param e  the engine
 *  @return always true (a missing block is said here and refused by the session) */
bool after_gpu(pulsar_engine *e) {
    uint32_t at = 0;
    const char *miss = pulsar_driver_first_missing(e->family->driver, &e->plan, &at);
    const pulsar_state_plan *sp = &e->state_plan;
    fprintf(stderr, "pulsar: %s: the core step driver over %u layers of shared blocks (N2 Mamba-2, F1 GQA, F2 router "
                    "512 / 22 + the rows MoE's latent relu^2 experts, E5 head); state: ssm FIXED DELTA + ssm_conv FIXED "
                    "SNAPSHOT + kv POOL%s%s%s\n", ARCH,
            pulsar_layer_plan_count(&e->plan, PULSAR_LAYER_BLOCKS),
            e->block_drafter     ? " (+ the block drafter's slots: draft_kv RING, draft_anchor FIXED)"
            : sp->n_slots > sp->n_trunk ? " (+ the drafter slot's kv POOL)"
                                        : "",
            miss ? "; MISSING " : "",
            miss ? miss : "");
    return true;
}

/** The engine's family state as Nemotron-H's (no family check: only this family's hooks call it).
 *  @param e  an engine this family loaded
 *  @return its pulsar_nemotron_h_model */
const pulsar_nemotron_h_model *model_of(const pulsar_engine *e) { return (const pulsar_nemotron_h_model *)e->family_state; }

/** pulsar_family::logits_width.
 *  @param e  the engine
 *  @return n_vocab (131072: every id real) */
uint32_t logits_width(const pulsar_engine *e) { return model_of(e)->shape.n_vocab; }
/** pulsar_family::model_name.
 *  @param e  the engine
 *  @return the shape's profile name (static) */
const char *model_name(const pulsar_engine *e) { return model_of(e)->shape.name; }
/** pulsar_family::served_model_id.
 *  @return "nemotron-3.5-super" (static) */
const char *served_model_id(const pulsar_engine *) { return "nemotron-3.5-super"; }
/* the ChatML + <think> renderer (src/lib/nemotron_chat.h): the engine's chat front (nemotron_h_tokenizer.cpp) and the
 * server's family table (src/server/family_nemotron_h.cpp) */
/** pulsar_family::chat_format.
 *  @return PULSAR_CHAT_NEMOTRON_H */
pulsar_chat_format chat_format(const pulsar_engine *) { return PULSAR_CHAT_NEMOTRON_H; }
/* Disk-KV compatibility id: 0 and 1 are DeepSeek's profiles, 2 Qwen's, 3 Inkling's. */
/** pulsar_family::model_id.
 *  @return 4 */
int model_id(const pulsar_engine *) { return 4; }
/* no positional encoding: the attention layers are trained to max_position_embeddings; the Mamba state is O(1) */
/** pulsar_family::trained_context.
 *  @param e  the engine
 *  @return the shape's max_context (max_position_embeddings), in tokens */
uint64_t trained_context(const pulsar_engine *e) { return model_of(e)->shape.max_context; }
/* N9: the MTP module drafts (the chained-MTP drafter, spec_mtp.h) when the container holds it */
/** pulsar_family::drafter.
 *  @param e  the engine
 *  @return MTP when the module drafts, BLOCK when LiLiCorr does, else NONE */
pulsar_drafter_kind drafter(pulsar_engine *e) {
    return model_of(e)->drafting ? PULSAR_DRAFTER_MTP : model_of(e)->block ? PULSAR_DRAFTER_BLOCK : PULSAR_DRAFTER_NONE;
}
/* the routed experts' tier in EXL3's value space (family.h quant_bits): 20 + the highest rate in half bits, 0 when
 * they are not EXL3 (BF16 today) */
/** pulsar_family::quant_bits: the highest EXL3 rate over the routed up / down stacks.
 *  @param e  the engine
 *  @return 20 + K in half bits, or 0 when the experts are not EXL3 */
int quant_bits(pulsar_engine *e) {
    const pulsar_nemotron_h_model *w = model_of(e);
    int bits = 0;
    for (uint32_t il = 0; il < w->shape.n_layer; il++)
        for (const pulsar_tensor *t : {w->layer[il].exp_up, w->layer[il].exp_down}) {
            const int k2 = t ? exl3_type_k2(t->type) : 0;
            if (k2 && 20 + k2 > bits) bits = 20 + k2;
        }
    return bits;
}

/* ---- the session: the core row-step driver (step_driver.cpp) over the family's ops (nemotron_h_forward.cpp) ----- */

/// The session's sync: the core driver's, over the declared state.
const pulsar_sync_ops k_sync = PULSAR_DRIVER_SYNC_OPS(PULSAR_NEMOTRON_H_ARCH);
/// The session table: the core row-step driver's ops (step_driver.cpp) over k_sync.
const pulsar_family_session_ops k_session_ops = PULSAR_DRIVER_SESSION_OPS(&k_sync);

}  // namespace

const pulsar_family PULSAR_FAMILY_NEMOTRON_H = {
    /* .id           = */ PULSAR_FAMILY_ID_NEMOTRON_H,
    /* .arch         = */ PULSAR_NEMOTRON_H_ARCH,
    /* .name         = */ "Nemotron-H",
    /* .caps         = */ PULSAR_FAMILY_CAP_CHAT | PULSAR_FAMILY_CAP_BANKS | PULSAR_FAMILY_CAP_SEGMENTS |
                          PULSAR_FAMILY_CAP_PAYLOAD | PULSAR_FAMILY_CAP_SPEC,
                          /* the tokenizer + chat front (N8); the driver's session ops over the declared state (N3,
                           * N7); SPEC: the driver's verify target under the MTP drafter (N9); then TP (N12), VISION
                           * (N11) */
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
    /* .spec         = */ &PULSAR_DRIVER_SPEC_TARGET,   /* N9: the core driver's verify target (step_driver.cpp) */
    /* .session      = */ &k_session_ops,
    /* .tokenizer    = */ &PULSAR_NEMOTRON_H_TOKENIZER,   /* nemotron_h_tokenizer.cpp (F5's BPE engine + renderer) */
    /* .tp_slices    = */ NULL,
    /* .act_kind     = */ PULSAR_ACT_KIND_ROWS,
    /* .vision       = */ NULL,   /* N11: the RADIO tower; v1 containers hold none */
    /* .banks        = */ &PULSAR_BANK_OPS_DECLARED,   /* state_kinds.cpp over the declared state */
    /* .imatrix      = */ NULL,
    /* .driver       = */ &PULSAR_NEMOTRON_H_DRIVER,   /* nemotron_h_forward.cpp */
    /* .unload       = */ unload,
};

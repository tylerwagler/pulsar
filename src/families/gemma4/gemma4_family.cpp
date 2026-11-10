/* gemma4_family.cpp -- the Gemma 4 family's descriptor (family.h): load, the engine facts, and the session on the
 * core row-step driver (gemma4_forward.cpp). */
/** @file gemma4_family.cpp
 *  @brief The Gemma 4 family descriptor: load, the engine facts, the session on the core row-step driver. */
#include "gemma4.h"
#include "engine/spec_block.h"
#include "lib/bpe_tokenizer.h"

#include <math.h>
#include <string.h>

namespace {

/// The architecture name every refusal and announce line is prefixed with.
const char *const ARCH = PULSAR_GEMMA4_ARCH;

/** pulsar_family::unload: free the tokenizer, the block drafter's record (L338 P3: a plain record, free() is its
 *  teardown) and the family state load allocated, and clear the engine's block-drafter pointer (safe after a partial
 *  load).
 *  @param e  the engine; its family_state is freed and set to NULL */
void unload(pulsar_engine *e) {
    if (e->family_state) {
        bpe_tokenizer_free(((pulsar_gemma4_model *)e->family_state)->tok);
        free(((pulsar_gemma4_model *)e->family_state)->block);
    }
    e->block_drafter = NULL;
    e->drafter_ops = NULL;
    free(e->family_state);
    e->family_state = NULL;
}

/** pulsar_family::load: refuse other families' options, then read the shape and plan, scan the artifact, bind and
 *  price every tensor.
 *  @param e    the engine; family_state and plan are written (family_state even on failure, for unload)
 *  @param opt  the open options
 *  @return true = loaded; false = refused (said, and counted through pulsar_load_refuse) */
bool load(pulsar_engine *e, const pulsar_engine_options *opt) {
    /* the options that name another family's feature refuse here, by name, rather than being ignored */
    if (opt->expert_overlay && opt->expert_overlay[0]) {
        fprintf(stderr, "pulsar: %s: --expert-overlay is a DeepSeek routed-expert tool; refusing\n", ARCH);
        return false;
    }
    if (opt->rope_yarn_factor != 0.0) {
        fprintf(stderr, "pulsar: %s: --rope-yarn: this family runs its trained ropes only; refusing\n", ARCH);
        return false;
    }
    if (opt->directional_steering_file && opt->directional_steering_file[0]) {
        fprintf(stderr, "pulsar: %s: directional steering is not implemented for this family; refusing\n", ARCH);
        return false;
    }
    pulsar_gemma4_model *w = (pulsar_gemma4_model *)xcalloc(1, sizeof(*w));
    e->family_state = w;   /* unload frees it, whatever this load's verdict */
    pulsar_load_refusals_reset();   /* the loader's one failure policy (family.cpp) */
    /* stage 1: the shape and the plan */
    if (!pulsar_gemma4_read_shape(&e->model, &w->shape) || pulsar_load_refusals()) {
        fprintf(stderr, "pulsar: %s: %u refusal(s) in the config -- the model does not load\n", ARCH,
                pulsar_load_refusals());
        return false;
    }
    pulsar_gemma4_plan(&w->shape, &e->plan);
    /* L338 P3: the block drafter the container holds (z-lab's plain DFlash), loaded unless --drafter none; --drafter
     * names nothing else here */
    {
        const char *kind = pulsar_block_drafter_kind_of(&e->model);
        if (!kind) return false;
        w->block_written = kind[0] != 0;
        const char *want = opt->drafter && opt->drafter[0] ? opt->drafter : NULL;
        if (want && strcmp(want, "none") && (!w->block_written || strcmp(want, kind))) {
            fprintf(stderr, "pulsar: %s: --drafter %s: the container holds %s%s -- refusing\n", ARCH, want,
                    w->block_written ? "the block drafter " : "no drafter", w->block_written ? kind : "");
            return false;
        }
        if (w->block_written && !(want && !strcmp(want, "none"))) {
            w->block = (pulsar_block_drafter *)xcalloc(1, sizeof(*w->block));
            pulsar_block_drafter *d = w->block;
            if (!pulsar_block_drafter_read(&e->model, d)) return false;
            if (d->head_kind != PULSAR_BLOCK_HEAD_PLAIN || d->hidden != w->shape.n_embd ||
                d->vocab != w->shape.n_vocab) {
                fprintf(stderr, "pulsar: %s: the drafter (%s, hidden %u, vocabulary %u) is not a plain DFlash over "
                                "this target's %u / %u -- refusing\n", ARCH, d->kind, d->hidden, d->vocab,
                        w->shape.n_embd, w->shape.n_vocab);
                return false;
            }
            /* the taps: HF decoder layer i's output = the residual after layer i's scaled add (its last sublayer) */
            for (uint32_t j = 0; j < d->n_taps; j++) {
                if (d->tap_block[j] >= w->shape.n_layer) {
                    fprintf(stderr, "pulsar: %s: the drafter taps layer %u of a %u-layer target -- refusing\n", ARCH,
                            d->tap_block[j], w->shape.n_layer);
                    return false;
                }
                d->tap_layer[j] = d->tap_block[j];
                d->tap_sub[j] = PULSAR_BLOCK_TAP_FFN;
            }
            /* E5: the target's embedding scale, bf16(sqrt(hidden)) (gemma4_forward.cpp head_decl), nearest even */
            const float r = sqrtf((float)w->shape.n_embd);
            uint32_t u;
            memcpy(&u, &r, 4);
            u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
            memcpy(&d->embed_scale, &u, 4);
        }
    }
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
    if (!pulsar_gemma4_bind(&e->model, w) || pulsar_load_refusals()) {
        fprintf(stderr, "pulsar: %s: %u refusal(s) binding the artifact -- the model does not load\n", ARCH,
                pulsar_load_refusals());
        return false;
    }
    /* stage 4: the tokenizer beside the shards */
    if (!pulsar_gemma4_load_tokenizer(w, opt->model_path)) return false;
    /* L338 P3: the drafter behind the round API -- the target's embedding and EXL3 head (PLAIN binds none of its own) */
    if (w->block) {
        w->block->embed = w->embed;
        w->block->head = w->head;
        w->block->head_rows = w->shape.n_vocab;
    }
    e->block_drafter = w->block;
    e->drafter_ops = w->block ? &w->block->ops : NULL;
    /* the session's declared state and the digest its disk segments are keyed by: the directory's names, types and
     * sizes */
    pulsar_gemma4_state_plan(e);
    if (!pulsar_state_plan_slots_fit(&e->state_plan, ARCH)) return false;   /* L375 O48: a drafter past the slot table */
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
    const pulsar_gemma4_shape *s = &w->shape;
    uint32_t n_full = 0;
    for (uint32_t il = 0; il < e->plan.n_layer; il++) n_full += e->plan.desc[il].mixer == PULSAR_MIXER_ATTN_FULL;
    fprintf(stderr, "pulsar: %s: %s, %u layers: %u window (%u keys, %uq/%ukv x %u) + %u global (%uq/%ukv x %u, K = V); "
                    "every layer a dense GeGLU MLP (%u, stored %u) + %u of %u experts (%u, stored %u); the pack's EXL3 head, "
                    "soft-cap %g; %.2f GiB bound\n",
            ARCH, s->name, s->n_layer, s->n_layer - n_full, s->window, s->n_head, s->swa_n_head_kv, s->swa_head_dim, n_full,
            s->n_head, s->n_head_kv, s->head_dim, s->n_ff, pulsar_gemma4_ff_stored(s), s->n_expert_used, s->n_expert,
            s->n_ff_exp, pulsar_gemma4_ff_exp_stored(s), (double)s->softcap,
            (double)w->census.bytes / (1024.0 * 1024.0 * 1024.0));
    return true;
}

/* The lane, announced once at open (rule 5): the core driver over the family's blocks, and what its table lacks for
 * the plan (a session refuses with exactly that). */
/** @param e  the engine
 *  @return always true (a missing block is said here and refused by the session) */
bool after_gpu(pulsar_engine *e) {
    uint32_t at = 0;
    const char *miss = pulsar_driver_first_missing(e->family->driver, &e->plan, &at);
    fprintf(stderr, "pulsar: %s: the core step driver over %u layers of shared blocks (F1 GQA window 256 / global 512 "
                    "K = V, the rows dense GeGLU MLP + F2 router 128 / 8 + the rows MoE's GeGLU experts, E5 head soft-cap); "
                    "state: kv_win RING + kv_full POOL%s%s\n", ARCH,
            pulsar_layer_plan_count(&e->plan, PULSAR_LAYER_BLOCKS), miss ? "; MISSING " : "", miss ? miss : "");
    return true;
}

/** The engine's family state as Gemma 4's (no family check: only this family's hooks call it).
 *  @param e  an engine this family loaded
 *  @return its pulsar_gemma4_model */
const pulsar_gemma4_model *model_of(const pulsar_engine *e) { return (const pulsar_gemma4_model *)e->family_state; }

/** pulsar_family::logits_width.
 *  @param e  the engine
 *  @return n_vocab (262144) */
uint32_t logits_width(const pulsar_engine *e) { return model_of(e)->shape.n_vocab; }
/** pulsar_family::model_name.
 *  @param e  the engine
 *  @return the shape's profile name (static) */
const char *model_name(const pulsar_engine *e) { return model_of(e)->shape.name; }
/** pulsar_family::served_model_id.
 *  @return "gemma-4-26b-a4b" (static) */
const char *served_model_id(const pulsar_engine *) { return "gemma-4-26b-a4b"; }
/** pulsar_family::chat_format: the Gemma 4 renderer (src/lib/gemma4_chat; the server's k_family_gemma4).
 *  @return PULSAR_CHAT_GEMMA4 */
pulsar_chat_format chat_format(const pulsar_engine *) { return PULSAR_CHAT_GEMMA4; }
/* Disk-KV compatibility id: 0 and 1 DeepSeek's profiles, 2 Qwen's, 3 Inkling's, 4 Nemotron-H's, 5 the encoders', 6
 * Parse's. */
/** pulsar_family::model_id.
 *  @return 7, the disk-KV compatibility id */
int model_id(const pulsar_engine *) { return 7; }
/** pulsar_family::trained_context.
 *  @param e  the engine
 *  @return the shape's max_context (max_position_embeddings), in tokens */
uint64_t trained_context(const pulsar_engine *e) { return model_of(e)->shape.max_context; }
/** pulsar_family::drafter: the DFlash block drafter when the container carries one (L338 P3).
 *  @param e  the engine
 *  @return PULSAR_DRAFTER_BLOCK with a block drafter loaded, else PULSAR_DRAFTER_NONE */
pulsar_drafter_kind drafter(pulsar_engine *e) {
    const pulsar_gemma4_model *w = (const pulsar_gemma4_model *)e->family_state;
    return w && w->block ? PULSAR_DRAFTER_BLOCK : PULSAR_DRAFTER_NONE;
}
/* the routed experts' tier in EXL3's value space (family.h quant_bits): 20 + the highest rate in half bits */
/** pulsar_family::quant_bits: the highest EXL3 rate over the routed expert stacks.
 *  @param e  the engine
 *  @return 20 + K in half bits */
int quant_bits(pulsar_engine *e) {
    const pulsar_gemma4_model *w = model_of(e);
    int bits = 0;
    for (uint32_t il = 0; il < w->shape.n_layer; il++)
        for (const pulsar_tensor *t : {w->layer[il].exp_gate, w->layer[il].exp_up, w->layer[il].exp_down}) {
            const int k2 = t ? exl3_type_k2(t->type) : 0;
            if (k2 && 20 + k2 > bits) bits = 20 + k2;
        }
    return bits;
}

/// The sync ops the core driver's session table runs over (step_driver.h).
const pulsar_sync_ops k_sync = PULSAR_DRIVER_SYNC_OPS(PULSAR_GEMMA4_ARCH);
/// The session table: the core row-step driver's ops (step_driver.cpp) over k_sync.
const pulsar_family_session_ops k_session_ops = PULSAR_DRIVER_SESSION_OPS(&k_sync);

}  // namespace

const pulsar_family PULSAR_FAMILY_GEMMA4 = {
    /* .id           = */ PULSAR_FAMILY_ID_GEMMA4,
    /* .arch         = */ PULSAR_GEMMA4_ARCH,
    /* .name         = */ "Gemma 4",
    /* .caps         = */ PULSAR_FAMILY_CAP_CHAT | PULSAR_FAMILY_CAP_BANKS | PULSAR_FAMILY_CAP_SEGMENTS |
                          PULSAR_FAMILY_CAP_PAYLOAD | PULSAR_FAMILY_CAP_SPEC,   /* the tokenizer + chat front (L338
                           * T3); SPEC: the driver's verify target under the block drafter (P3) */
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
    /* .spec         = */ &PULSAR_DRIVER_SPEC_TARGET,   /* L338 P3: the core driver's verify target (step_driver.cpp) */
    /* .session      = */ &k_session_ops,
    /* .tokenizer    = */ &PULSAR_GEMMA4_TOKENIZER,   /* gemma4_tokenizer.cpp (the shared BPE engine's SentencePiece-BPE
                                                     row + src/lib/gemma4_chat's front) */
    /* .tp_slices    = */ NULL,
    /* .act_kind     = */ PULSAR_ACT_KIND_ROWS,
    /* .vision       = */ NULL,
    /* .banks        = */ &PULSAR_BANK_OPS_DECLARED,   /* state_kinds.cpp over the declared state */
    /* .imatrix      = */ NULL,
    /* .driver       = */ &PULSAR_GEMMA4_DRIVER,   /* gemma4_forward.cpp */
    /* .unload       = */ unload,
};

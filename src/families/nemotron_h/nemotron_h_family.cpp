/* nemotron_h_family.cpp -- the Nemotron-H family's descriptor (family.h): load, the engine facts, and the session --
 * the core row-step driver over the family's blocks (nemotron_h_forward.cpp) and its declared state (state_kinds.h). */
#include "nemotron_h.h"
#include "engine/exl3_trellis.h"
#include "lib/bpe_tokenizer.h"

#include <algorithm>

namespace {

const char *const ARCH = PULSAR_NEMOTRON_H_ARCH;

void unload(pulsar_engine *e) {
    if (pulsar_nemotron_h_model *w = (pulsar_nemotron_h_model *)e->family_state) {
        bpe_tokenizer_free(w->tok);
        pulsar_gpu_tensor_free(w->draft_head);
        free(w->draft_ids);
    }
    free(e->family_state);
    e->family_state = NULL;
}

#include "nemotron_h_draft_vocab.inc"

/* L296 draft-cost: the drafter's head's ids (pulsar_nemotron_h_model::draft_ids, ascending) -- the first `rows` ids of
 * nemotron_h_draft_vocab.inc's list (the named added tokens, then the ids most frequent first; tools/nemotron/
 * draft_vocab.py).  The list was ranked under this checkpoint's tokenizer: the digest of its ids' token bytes is
 * recomputed here and a mismatch refuses (it would rank another tokenizer's tokens). */
bool draft_vocab(pulsar_nemotron_h_model *w) {
    const uint32_t n_list = (uint32_t)(sizeof k_nemotron_draft_ids / sizeof k_nemotron_draft_ids[0]);
    uint64_t h = 1469598103934665603ull;
    bool ok = true;
    for (uint32_t i = 0; ok && i < n_list; i++) {
        size_t len = 0;
        const int32_t id = k_nemotron_draft_ids[i];
        const char *b = (uint32_t)id < w->shape.n_vocab ? bpe_tokenizer_token_bytes(w->tok, id, &len) : NULL;
        ok = b != NULL;
        for (size_t j = 0; ok && j <= len; j++) {
            h ^= j < len ? (uint8_t)b[j] : 0xffu;
            h *= 1099511628211ull;
        }
    }
    if (!ok || h != NEMOTRON_DRAFT_VOCAB_DIGEST) {
        fprintf(stderr, "pulsar: %s: the MTP draft vocabulary (nemotron_h_draft_vocab.inc) was ranked under another "
                        "tokenizer (digest %016llx, want %016llx) -- regenerate it with tools/nemotron/draft_vocab.py; "
                        "refusing\n", ARCH, (unsigned long long)h, (unsigned long long)NEMOTRON_DRAFT_VOCAB_DIGEST);
        return false;
    }
    /* PULSAR_NEMOTRON_MTP_DRAFT_ROWS: how many of the list's ids the head carries (the sweep's knob; proposals only) */
    uint32_t rows = NEMOTRON_DRAFT_ROWS_DEFAULT;
    if (const char *ev = getenv("PULSAR_NEMOTRON_MTP_DRAFT_ROWS"); ev && ev[0]) rows = (uint32_t)strtoul(ev, NULL, 10);
    if (rows < NEMOTRON_DRAFT_FIXED || rows > n_list) {
        fprintf(stderr, "pulsar: %s: a draft head of %u rows (PULSAR_NEMOTRON_MTP_DRAFT_ROWS) is outside the list's "
                        "[%u, %u]; refusing\n", ARCH, rows, NEMOTRON_DRAFT_FIXED, n_list);
        return false;
    }
    w->draft_ids = (int32_t *)xmalloc((size_t)rows * sizeof(int32_t));
    memcpy(w->draft_ids, k_nemotron_draft_ids, (size_t)rows * sizeof(int32_t));
    std::sort(w->draft_ids, w->draft_ids + rows);
    w->n_draft = rows;
    return true;
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
    /* N9: the MTP module drafts unless the open turned drafting off (the option every family's drafter honours) */
    w->drafting = w->shape.mtp_written && !opt->dspark_disable;
    e->drafter_ops = w->drafting ? &PULSAR_NEMOTRON_H_DRAFTER : NULL;
    if (w->drafting && !draft_vocab(w)) return false;
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
                        : s->mtp_written ? "bound, drafting off" : "not in the container");
    return true;
}

/* The lane, announced once at open (rule 5): the core driver over the family's blocks, and what its table lacks for
 * the plan (a session refuses with exactly that). */
bool after_gpu(pulsar_engine *e) {
    /* L296 draft-cost: the drafter's head, built once from the head's rows of the draft vocabulary */
    pulsar_nemotron_h_model *w = (pulsar_nemotron_h_model *)e->family_state;
    if (w->drafting) {
        w->draft_head = pulsar_draft_head_build(&e->model, w->head, (int)w->shape.n_embd, (int)w->shape.n_vocab,
                                                w->draft_ids, w->n_draft, "nemotron_h MTP");
        if (!w->draft_head) return false;
    }
    uint32_t at = 0;
    const char *miss = pulsar_driver_first_missing(e->family->driver, &e->plan, &at);
    const pulsar_state_plan *sp = &e->state_plan;
    fprintf(stderr, "pulsar: %s: the core step driver over %u layers of shared blocks (N2 Mamba-2, F1 GQA, F2 router "
                    "512 / 22 + the rows MoE's latent relu^2 experts, E5 head); state: ssm FIXED DELTA + ssm_conv FIXED "
                    "SNAPSHOT + kv POOL%s%s%s\n", ARCH,
            pulsar_layer_plan_count(&e->plan, PULSAR_LAYER_BLOCKS),
            sp->n_slots > sp->n_trunk ? " (+ the drafter slot's kv POOL)" : "", miss ? "; MISSING " : "",
            miss ? miss : "");
    return true;
}

const pulsar_nemotron_h_model *model_of(const pulsar_engine *e) { return (const pulsar_nemotron_h_model *)e->family_state; }

uint32_t logits_width(const pulsar_engine *e) { return model_of(e)->shape.n_vocab; }
const char *model_name(const pulsar_engine *e) { return model_of(e)->shape.name; }
const char *served_model_id(const pulsar_engine *) { return "nemotron-3.5-super"; }
/* the ChatML + <think> renderer (src/lib/nemotron_chat.h): the engine's chat front (nemotron_h_tokenizer.cpp) and the
 * server's family table (src/server/family_nemotron_h.cpp) */
pulsar_chat_format chat_format(const pulsar_engine *) { return PULSAR_CHAT_NEMOTRON_H; }
/* Disk-KV compatibility id: 0 and 1 are DeepSeek's profiles, 2 Qwen's, 3 Inkling's. */
int model_id(const pulsar_engine *) { return 4; }
/* no positional encoding: the attention layers are trained to max_position_embeddings; the Mamba state is O(1) */
uint64_t trained_context(const pulsar_engine *e) { return model_of(e)->shape.max_context; }
/* N9: the MTP module drafts (the chained-MTP drafter, spec_mtp.h) when the container holds it */
pulsar_drafter_kind drafter(pulsar_engine *e) { return model_of(e)->drafting ? PULSAR_DRAFTER_MTP : PULSAR_DRAFTER_NONE; }
/* the routed experts' tier in EXL3's value space (family.h quant_bits): 20 + the highest rate in half bits, 0 when
 * they are not EXL3 (BF16 today) */
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

const pulsar_sync_ops k_sync = PULSAR_DRIVER_SYNC_OPS(PULSAR_NEMOTRON_H_ARCH);
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

/* spec_block.cpp -- the block drafter (spec_block.h; L298): its declaration and binding from the container, its
 * declared state, the context absorb, the block forward and the chain, behind the speculation round API
 * (session_spec.cpp) and the core driver's verify target (step_driver.cpp PULSAR_DRIVER_SPEC_TARGET). */
#include "pulsar_engine_internal.h"
#include "spec_block.h"
#include "spec_mtp.h"
#include "cuda/pulsar_cuda_attn_gqa.h"
#include "cuda/pulsar_cuda_blockdraft.h"
#include "cuda/pulsar_cuda_candsel.h"
#include "cuda/pulsar_cuda_head.h"
#include "cuda/pulsar_cuda_lilicorr.h"
#include "cuda/pulsar_cuda_qwen.h"     /* the rows front door (pulsar_rows_linear_*) */
#include "cuda/pulsar_cuda_rowops.h"

#include <math.h>

#include <algorithm>
#include <vector>

namespace {

const char *const OWNER = "block drafter";
const char *const BLOCK = "drafter";

bool refuse(const char *fmt, const char *a = "", const char *b = "") {
    fprintf(stderr, "pulsar: block drafter: ");
    fprintf(stderr, fmt, a, b);
    fprintf(stderr, " -- refusing\n");
    pulsar_load_refuse();
    return false;
}
bool fail(const char *what) {
    fprintf(stderr, "pulsar: block drafter: %s -- refusing the step\n", what);
    return false;
}
uint64_t a256(uint64_t x) { return (x + 255u) & ~(uint64_t)255u; }
void *dptr(pulsar_gpu_tensor *t) { return pulsar_gpu_tensor_device_ptr(t); }

const pulsar_block_drafter *DR(const pulsar_engine *e) { return e->block_drafter; }

/* ---- the declaration ------------------------------------------------------------------------------------------- */

bool get_u32(const pulsar_model *m, const char *key, uint32_t *out) {
    uint64_t v = 0;
    if (!model_get_u64_compat(m, key, &v) || v > 0xffffffffull) return refuse("%s is missing or not an unsigned integer%s", key);
    *out = (uint32_t)v;
    return true;
}
bool get_f32(const pulsar_model *m, const char *key, float *out) {
    if (!model_get_f32_compat(m, key, out) || !isfinite(*out)) return refuse("%s is missing or not a finite number%s", key);
    return true;
}
bool want_bool(const pulsar_model *m, const char *key, bool want) {
    bool v = !want;
    if (!model_get_bool(m, key, &v)) return refuse("%s is missing or not a bool%s", key);
    if (v != want) return refuse("%s is %s: the block drafter implements the other value", key, v ? "true" : "false");
    return true;
}

/* an array of unsigned block ids */
bool get_ids(const pulsar_model *m, const char *key, uint32_t *out, uint32_t cap, uint32_t *n) {
    pulsar_array_ref arr;
    if (!model_get_array(m, key, &arr) ||
        (arr.type != PULSAR_META_UINT32 && arr.type != PULSAR_META_INT32 && arr.type != PULSAR_META_UINT64 &&
         arr.type != PULSAR_META_INT64) || arr.len == 0 || arr.len > cap)
        return refuse("%s is missing, not an integer array, or longer than the taps a block drafter holds%s", key);
    pulsar_cursor c = cursor_at(m, arr.data_pos);
    for (uint64_t i = 0; i < arr.len; i++) {
        int64_t v = -1;
        bool ok;
        if (arr.type == PULSAR_META_UINT64 || arr.type == PULSAR_META_INT64) {
            uint64_t u = 0;
            ok = cursor_u64(&c, &u);
            v = (int64_t)u;
        } else {
            uint32_t u = 0;
            ok = cursor_u32(&c, &u);
            v = arr.type == PULSAR_META_INT32 ? (int64_t)(int32_t)u : (int64_t)u;
        }
        if (!ok || v < 0 || v > 0xffff) return refuse("%s holds an entry that is not a block id%s", key);
        out[i] = (uint32_t)v;
    }
    *n = (uint32_t)arr.len;
    return true;
}

}  // namespace

namespace {

/* THE KIND TABLE (spec_block.h THE DECLARATION): per kind, where its config keeps each number (keys under
 * "<kind>."), the chain head it ends in and its stop rule's tau. */
struct kind_decl {
    const char *kind;
    pulsar_block_head_kind head;
    float tau;
    const char *hidden, *n_layer, *n_head, *n_kv, *head_dim, *ff, *vocab, *block, *eps, *rope_theta;
    const char *mask, *conv_taps, *conv_group, *window, *topk, *taps;
    const char *non_causal;   /* a bool key that must be false: bidirectional inside the block */
    const char *swa;          /* a bool key that must be true: every layer windowed over the context */
};

/* LiLiCorr (L298): vLLM's lilicorr config; tau measured (L298, the l297 bench single stream T0, the priced width on):
 * tau off 62.0 tok/s, 0.3 62.0, 0.5 62.6 (code 61.1 -> 64.2), 0.7 61.4.
 * DFlash2 (L299): z-lab's DFlash2DraftModel config (incoai/Qwen3.8-Flash-Next-DFlash2); tau measured (L299, the
 * scoreboard untraced, Qwen3.8 U-e4-d5, the priced width on; code / prose / code T0.7 / edit / 4 streams tok/s): 0.3
 * 122.2 / 81.2 / 108.5 / 128.6 / 171.9, 0.5 121.6 / 83.1 / 107.2 / 129.3 / 175.7, 0.7 117.1 / 81.8 / 106.1 / 127.9
 * / 174.3.  Its slot confidence is calibrated (the served-round replay, quintiles conf -> accept: prose 0.44 -> 0.30,
 * 0.73 -> 0.54, 0.93 -> 0.79, 0.99 -> 0.97; code 0.68 -> 0.60, 0.97 -> 0.93, 1.00 -> 0.99): no rescale. */
const kind_decl k_kinds[] = {
    {"lilicorr", PULSAR_BLOCK_HEAD_LILICORR, 0.5f, "hidden_size", "num_hidden_layers",
     "num_attention_heads", "num_key_value_heads", "head_dim", "intermediate_size", "vocab_size", "block_size",
     "rms_norm_eps", "rope_theta", "dflash.mask_token_id", "dflash.conv_kernel_size", "dflash.conv_group_size",
     "dflash.swa_window_size", "dflash.lilicorr_candidate_topk", "dflash.target_layer_ids", "dflash.causal",
     "dflash.use_swa"},
    {"dflash2", PULSAR_BLOCK_HEAD_SELECTOR, 0.5f, "hidden_size", "num_hidden_layers",
     "num_attention_heads", "num_key_value_heads", "head_dim", "intermediate_size", "vocab_size",
     "dflash_config.block_size", "rms_norm_eps", "rope_parameters.rope_theta", "dflash_config.mask_token_id",
     "dflash_config.conv_kernel_size", "dflash_config.conv_group_size", "sliding_window",
     "dflash_config.selector_top_k", "dflash_config.target_layer_ids", "is_causal", "use_sliding_window"},
};
const kind_decl *kind_find(const char *kind) {
    for (const kind_decl &k : k_kinds)
        if (!strcmp(k.kind, kind)) return &k;
    return NULL;
}

/* "<kind>.<key>" */
struct kkey {
    char buf[160];
    kkey(const char *kind, const char *key) { snprintf(buf, sizeof buf, "%s.%s", kind, key); }
    operator const char *() const { return buf; }
};

}  // namespace

const char *pulsar_block_drafter_kind_of(const pulsar_model *m) {
    bool present = false;
    for (uint64_t i = 0; i < m->n_kv && !present; i++) present = pulsar_streq(m->kv[i].key, "pulsar.drafter");
    if (!present) return "";
    pulsar_str v = {NULL, 0};
    if (!model_get_string(m, "pulsar.drafter", &v)) {
        refuse("pulsar.drafter is not a string%s%s");
        return NULL;
    }
    if (v.len == 0) return "";
    for (const kind_decl &k : k_kinds)
        if (pulsar_streq(v, k.kind)) return k.kind;
    fprintf(stderr, "pulsar: block drafter: pulsar.drafter names %.*s, a drafter the lane does not implement (",
            (int)v.len, v.ptr);
    for (const kind_decl &k : k_kinds) fprintf(stderr, "%s%s", &k == k_kinds ? "" : ", ", k.kind);
    fprintf(stderr, ") -- refusing\n");
    pulsar_load_refuse();
    return NULL;
}

const char *const *pulsar_block_drafter_blocks(const char *kind) {
    static const char *const lilicorr[] = {BLOCK, "lilicorr", NULL};
    static const char *const selector[] = {BLOCK, "selector", NULL};
    const kind_decl *k = kind_find(kind);
    return k && k->head == PULSAR_BLOCK_HEAD_LILICORR ? lilicorr : selector;
}

bool pulsar_block_drafter_read(const pulsar_model *m, pulsar_block_drafter *d) {
    *d = pulsar_block_drafter{};
    const char *kind = pulsar_block_drafter_kind_of(m);
    if (!kind) return false;
    const kind_decl *K = kind[0] ? kind_find(kind) : NULL;
    if (!K) return refuse("the container declares no block drafter (pulsar.drafter)%s%s");
    d->kind = K->kind;
    d->head_kind = K->head;
    const char *kd = K->kind;
    bool ok = get_u32(m, kkey(kd, K->hidden), &d->hidden);
    ok = get_u32(m, kkey(kd, K->n_layer), &d->n_layer) && ok;
    ok = get_u32(m, kkey(kd, K->n_head), &d->n_head) && ok;
    ok = get_u32(m, kkey(kd, K->n_kv), &d->n_kv) && ok;
    ok = get_u32(m, kkey(kd, K->head_dim), &d->head_dim) && ok;
    ok = get_u32(m, kkey(kd, K->ff), &d->ff) && ok;
    ok = get_u32(m, kkey(kd, K->vocab), &d->vocab) && ok;
    ok = get_u32(m, kkey(kd, K->block), &d->block) && ok;
    ok = get_f32(m, kkey(kd, K->eps), &d->eps) && ok;
    ok = get_f32(m, kkey(kd, K->rope_theta), &d->rope_theta) && ok;
    uint32_t mask = 0;
    ok = get_u32(m, kkey(kd, K->mask), &mask) && ok;
    d->mask_id = (int32_t)mask;
    ok = get_u32(m, kkey(kd, K->conv_taps), &d->conv_taps) && ok;
    ok = get_u32(m, kkey(kd, K->conv_group), &d->conv_group) && ok;
    ok = get_u32(m, kkey(kd, K->window), &d->window) && ok;
    ok = get_u32(m, kkey(kd, K->topk), &d->topk) && ok;
    ok = want_bool(m, kkey(kd, K->non_causal), false) && ok;   /* bidirectional inside the block */
    ok = want_bool(m, kkey(kd, K->swa), true) && ok;           /* every layer windowed over the context */
    ok = get_ids(m, kkey(kd, K->taps), d->tap_block, PULSAR_BLOCK_MAX_TAPS, &d->n_taps) && ok;
    if (!ok) return false;
    /* the numbers the blocks implement (the shared GQA block's tiles, the round record, the top-k kernel) */
    const char *why = d->n_layer == 0 || d->n_layer > PULSAR_BLOCK_MAX_LAYERS ? "1 .. 8 layers"
                      : d->head_dim != 64u && d->head_dim != 128u          ? "a head_dim the GQA block tiles (64, 128)"
                      : d->n_kv == 0 || d->n_head % d->n_kv || d->n_head / d->n_kv > 64u
                          ? "n_head a multiple of n_kv, the group <= 64"
                      : d->hidden % 64u || d->ff % 64u                     ? "hidden and ff multiples of 64"
                      : d->block < 2u || d->block > PULSAR_BLOCK_SIZE_MAX   ? "a block of 2 .. 16 rows"
                      : d->conv_taps == 0 || d->conv_taps > d->block       ? "1 .. block conv taps"
                      : d->conv_group == 0 || d->hidden % d->conv_group    ? "a conv group dividing hidden"
                      : d->window == 0                                     ? "a sliding window"
                      : d->topk == 0 || d->topk > PULSAR_BLOCKDRAFT_TOPK_MAX ? "1 .. 16 candidates a slot"
                      : !(d->eps > 0.f) || !(d->rope_theta > 0.f)          ? "a positive eps and rope base"
                      : d->mask_id < 0 || (uint32_t)d->mask_id >= d->vocab ? "a mask token inside the vocabulary"
                                                                           : NULL;
    if (why) return refuse("the declaration (%s.*) is outside what the blocks implement: %s", kd, why);
    if (d->head_kind == PULSAR_BLOCK_HEAD_LILICORR) {
        /* the candidate head: the kernel's one geometry (pulsar_cuda_lilicorr.h) */
        uint32_t lh = 0, ll = 0, lnh = 0, lf = 0;
        float ratio = 0.f;
        ok = get_u32(m, kkey(kd, "dflash.lilicorr_hidden_size"), &lh);
        ok = get_u32(m, kkey(kd, "dflash.lilicorr_num_layers"), &ll) && ok;
        ok = get_u32(m, kkey(kd, "dflash.lilicorr_num_heads"), &lnh) && ok;
        ok = get_u32(m, kkey(kd, "dflash.lilicorr_factor_dim"), &lf) && ok;
        ok = get_f32(m, kkey(kd, "dflash.lilicorr_mlp_ratio"), &ratio) && ok;
        ok = get_f32(m, kkey(kd, "dflash.lilicorr_vector_eps"), &d->head_lc.vector_eps) && ok;
        ok = get_f32(m, kkey(kd, "dflash.lilicorr_logit_scale"), &d->head_lc.logit_scale) && ok;
        if (!ok) return false;
        if (lh != PULSAR_LILICORR_HIDDEN || ll != PULSAR_LILICORR_LAYERS || lnh != PULSAR_LILICORR_HEADS ||
            lf != PULSAR_LILICORR_HIDDEN || ratio != 2.0f || d->topk != PULSAR_LILICORR_TOPK ||
            d->block != PULSAR_LILICORR_SLOTS + 1u || d->hidden != PULSAR_LILICORR_MODEL)
            return refuse("the candidate head's geometry (%s.dflash.lilicorr_*) is not the head kernel's (1024 wide, 2 "
                          "layers x 8 heads, factor 1024, mlp x2, 8 candidates at 7 slots over a 4096 drafter)%s", kd);
    } else {
        /* the selector: the kernel's caps (pulsar_cuda_candsel.h) */
        if (!get_u32(m, kkey(kd, "dflash_config.selector_rank"), &d->sel.rank)) return false;
        if (d->sel.rank == 0 || d->sel.rank > PULSAR_CANDSEL_RANK_MAX || d->topk > PULSAR_CANDSEL_TOPK_MAX ||
            d->block - 1u > PULSAR_CANDSEL_SLOTS_MAX)
            return refuse("the selector (%s.dflash_config.selector_*) is outside the kernel's: rank 1 .. 256, 16 "
                          "candidates, 15 slots%s", kd);
    }
    d->ops = PULSAR_BLOCK_DRAFTER_OPS;
    d->ops.depth = d->ops.depth_max = d->block - 1u;   /* a chain is the block's mask slots */
    d->ops.tau = K->tau;
    return true;
}

bool pulsar_block_drafter_bind(pulsar_role_binder *b, pulsar_block_drafter *d) {
    const pulsar_role_admit A_BF16 = {PULSAR_ROLE_PLAIN(PULSAR_TENSOR_BF16), false, PULSAR_ROLE_DENSE, 0};
    const pulsar_role_admit A_I32 = {PULSAR_ROLE_PLAIN(PULSAR_TENSOR_I32), false, PULSAR_ROLE_DENSE, 0};
    const pulsar_role_admit A_LINEAR = {0, true, PULSAR_ROLE_DENSE, PULSAR_ACTS(PULSAR_ACT_ROWS_BF16)};
    const uint64_t E = d->hidden, Q = (uint64_t)d->n_head * d->head_dim, KV = (uint64_t)d->n_kv * d->head_dim;
    const uint64_t CP = 2ull * d->conv_taps * (E / d->conv_group);
    auto top = [&](const char *role) { return pulsar_role_in(role, BLOCK, -1); };
    d->fc = pulsar_role_bind(b, top("fc"), A_LINEAR, 2, (uint64_t)d->n_taps * E, E);
    d->hidden_norm = pulsar_role_bind(b, top("hidden_norm"), A_BF16, 1, E);
    d->out_norm = pulsar_role_bind(b, top("out_norm"), A_BF16, 1, E);
    /* the head: the whole vocabulary, or (L299) a subset of it whose rows' ids are the container's head_ids */
    d->head_rows = d->vocab;
    if (const pulsar_tensor *ids = pulsar_role_binder_find(b, top("head_ids")))
        d->head_ids = pulsar_role_bind(b, top("head_ids"), A_I32, 1, ids->ndim == 1 ? ids->dim[0] : 0);
    if (d->head_ids) {
        d->head_rows = (uint32_t)d->head_ids->dim[0];
        if (d->head_kind != PULSAR_BLOCK_HEAD_SELECTOR || d->head_rows == 0 || d->head_rows > d->vocab ||
            d->head_rows < d->topk) {
            refuse("drafter head_ids: %s%s", d->head_kind != PULSAR_BLOCK_HEAD_SELECTOR
                                                 ? "the LiLiCorr head gathers the whole vocabulary's candidates"
                                                 : "a subset of the vocabulary at least topk rows long");
            d->head_ids = NULL;
        }
    }
    d->head = pulsar_role_bind(b, top("head"), A_LINEAR, 2, E, d->head_rows);
    for (uint32_t l = 0; l < d->n_layer; l++) {
        pulsar_block_layer &L = d->layer[l];
        auto k = [&](const char *role) { return pulsar_role_in(role, BLOCK, (int32_t)l); };
        L.in_norm = pulsar_role_bind(b, k("in_norm"), A_BF16, 1, E);
        L.post_norm = pulsar_role_bind(b, k("post_norm"), A_BF16, 1, E);
        L.q = pulsar_role_bind(b, k("attn_q"), A_LINEAR, 2, E, Q);
        L.k = pulsar_role_bind(b, k("attn_k"), A_LINEAR, 2, E, KV);
        L.v = pulsar_role_bind(b, k("attn_v"), A_LINEAR, 2, E, KV);
        L.o = pulsar_role_bind(b, k("attn_o"), A_LINEAR, 2, Q, E);
        L.q_norm = pulsar_role_bind(b, k("q_norm"), A_BF16, 1, d->head_dim);
        L.k_norm = pulsar_role_bind(b, k("k_norm"), A_BF16, 1, d->head_dim);
        L.ac_base = pulsar_role_bind(b, k("attn_conv_base"), A_BF16, 3, E, d->conv_taps, 2);
        L.ac_proj = pulsar_role_bind(b, k("attn_conv_proj"), A_LINEAR, 2, E, CP);
        L.mc_base = pulsar_role_bind(b, k("mlp_conv_base"), A_BF16, 3, E, d->conv_taps, 2);
        L.mc_proj = pulsar_role_bind(b, k("mlp_conv_proj"), A_LINEAR, 2, E, CP);
        L.gate = pulsar_role_bind(b, k("mlp_gate"), A_LINEAR, 2, E, d->ff);
        L.up = pulsar_role_bind(b, k("mlp_up"), A_LINEAR, 2, E, d->ff);
        L.down = pulsar_role_bind(b, k("mlp_down"), A_LINEAR, 2, d->ff, E);
    }
    if (d->head_kind == PULSAR_BLOCK_HEAD_SELECTOR) {
        /* the selector, block `selector`: its projection a Linear, the codebooks bf16 rows (gathered, never streamed) */
        pulsar_block_selector &S = d->sel;
        auto sk = [&](const char *role) { return pulsar_role_in(role, "selector", -1); };
        S.proj = pulsar_role_bind(b, sk("hidden_proj"), A_LINEAR, 2, E, S.rank);
        S.pred = pulsar_role_bind(b, sk("pred_codebook"), A_BF16, 2, S.rank, d->vocab);
        S.succ = pulsar_role_bind(b, sk("succ_codebook"), A_BF16, 2, S.rank, d->vocab);
        return d->fc && d->hidden_norm && d->out_norm && d->head && S.proj && S.pred && S.succ;
    }
    /* the candidate head, block `lilicorr`: every tensor as stored (the builder's ne order: [in, out] Linears) */
    pulsar_block_head &H = d->head_lc;
    const uint64_t LH = PULSAR_LILICORR_HIDDEN, S = PULSAR_LILICORR_SLOTS, K = PULSAR_LILICORR_TOPK,
                   NH = PULSAR_LILICORR_HEADS;
    auto hk = [&](const char *role, int32_t l = -1) { return pulsar_role_in(role, "lilicorr", l); };
    auto vec = [&](const char *role, uint64_t n, int32_t l = -1) { return pulsar_role_bind(b, hk(role, l), A_BF16, 1, n, 0, 0); };
    auto lin = [&](const char *role, uint64_t in, uint64_t out, int32_t l = -1) {
        return pulsar_role_bind(b, hk(role, l), A_LINEAR, 2, in, out);
    };
    H.token_proj = lin("token_proj", E, LH);
    H.token_proj_b = vec("token_proj_bias", LH);
    H.pass_proj = lin("pass_hidden_proj", E, LH);
    H.pass_proj_b = vec("pass_hidden_proj_bias", LH);
    H.context_proj = lin("context_proj", E, LH);
    H.context_proj_b = vec("context_proj_bias", LH);
    H.feature_norm_w = vec("feature_norm", 5);
    H.feature_norm_b = vec("feature_norm_bias", 5);
    H.feature_up_w = pulsar_role_bind(b, hk("feature_up"), A_BF16, 2, 5, LH);
    H.feature_up_b = vec("feature_up_bias", LH);
    H.feature_down = lin("feature_down", LH, LH);
    H.feature_down_b = vec("feature_down_bias", LH);
    const uint64_t slot_dims[5] = {LH, 1, S, 1, 1}, rank_dims[5] = {LH, K, 1, 1, 1};
    H.slot_embedding = pulsar_role_bind_n(b, hk("slot_embedding"), A_BF16, 5, slot_dims);
    H.rank_embedding = pulsar_role_bind_n(b, hk("rank_embedding"), A_BF16, 5, rank_dims);
    H.relative_slot_bias = pulsar_role_bind(b, hk("relative_slot_bias"), A_BF16, 2, 2 * S + 1, NH);
    H.same_slot_bias = vec("same_slot_bias", NH);
    H.output_norm = vec("output_norm", LH);
    H.anchor_norm = vec("anchor_norm", LH);
    H.factor_in = lin("factor_in", 3 * LH, LH);
    H.factor_in_b = vec("factor_in_bias", LH);
    H.out_head = lin("out_head", LH, LH);
    H.out_head_b = vec("out_head_bias", LH);
    H.in_head = lin("in_head", LH, LH);
    H.in_head_b = vec("in_head_bias", LH);
    H.anchor_out_head = lin("anchor_out_head", LH, LH);
    H.anchor_out_head_b = vec("anchor_out_head_bias", LH);
    for (int32_t l = 0; l < (int32_t)PULSAR_LILICORR_LAYERS; l++) {
        auto &Y = H.layer[l];
        Y.attn_norm = vec("attn_norm", LH, l);
        Y.attn_in = lin("attn_in", LH, 3 * LH, l);
        Y.attn_in_b = vec("attn_in_bias", 3 * LH, l);
        Y.attn_out = lin("attn_out", LH, LH, l);
        Y.attn_out_b = vec("attn_out_bias", LH, l);
        Y.mlp_norm = vec("mlp_norm", LH, l);
        Y.mlp_up = lin("mlp_up", LH, 2 * LH, l);
        Y.mlp_up_b = vec("mlp_up_bias", 2 * LH, l);
        Y.mlp_down = lin("mlp_down", 2 * LH, LH, l);
        Y.mlp_down_b = vec("mlp_down_bias", LH, l);
    }
    return d->fc && d->hidden_norm && d->out_norm && d->head;
}

namespace {

/* the GQA block's declaration of drafter layer l: n_head / n_kv / head_dim, the window, 1/sqrt(head_dim), the per-head
 * q / k RMSNorm with the layer's weights, the NeoX rope; `block` for the query block (bidirectional, never stored), 0
 * for the context store */
pulsar_gqa_layer gqa_decl(const pulsar_block_drafter *d, const uint16_t *qn, const uint16_t *kn, bool block) {
    pulsar_gqa_layer L{};
    L.n_q = d->n_head;
    L.n_kv = d->n_kv;
    L.head_dim = d->head_dim;
    /* E4M3 K / V at unit scale, every kind: LiLiCorr was served (and its acceptance measured) with vLLM's fp8_e4m3 KV
     * cache, whose rounding cuts the attention's distance to it 4x against bf16 rows (L298 head grade); DFlash2's
     * reference keeps bf16, and E4M3 rows measured no different on its served-acceptance replay (L299, every Linear
     * nvfp4: tokens / round code 5.80 -> 5.86, prose 3.24 -> 3.26, story 3.12 -> 3.12, edit 7.60 -> 7.60).  Half the
     * ring, and the ring rides every grid checkpoint. */
    L.kv_fmt = PULSAR_GQA_KV_E4M3;
    L.span = PULSAR_GQA_WINDOW;
    L.window = d->window;
    L.scale = (float)(1.0 / sqrt((double)d->head_dim));
    L.qk_norm = 1;
    L.norm_eps = d->eps;
    L.q_norm_w = qn;
    L.k_norm_w = kn;
    L.rope_theta = d->rope_theta;
    L.block = block ? 1 : 0;
    return L;
}

}  // namespace

void pulsar_block_drafter_plan(pulsar_engine *e, pulsar_state_plan *p) {
    pulsar_block_drafter *d = e->block_drafter;
    const pulsar_gqa_layer g = gqa_decl(d, NULL, NULL, false);
    pulsar_state_decl kv{"draft_kv", PULSAR_STATE_RING, PULSAR_STATE_CAPTURE_NONE, false, false, d->window,
                         pulsar_gqa_kv_token_bytes(&g), 0, 0, NULL, NULL, NULL};
    d->decl_kv = pulsar_state_plan_add(p, kv);
    for (uint32_t l = 0; l < d->n_layer; l++) p->slot[p->n_slots + l] = 1u << d->decl_kv;
    d->decl_anchor = UINT32_MAX;
    if (d->head_kind == PULSAR_BLOCK_HEAD_LILICORR) {   /* the candidate head reads cn of the last committed row */
        pulsar_state_decl anchor{"draft_anchor", PULSAR_STATE_FIXED, PULSAR_STATE_CAPTURE_NONE, false, false, 0,
                                 (uint64_t)d->hidden * 2u, 0, 0, NULL, NULL, NULL};
        d->decl_anchor = pulsar_state_plan_add(p, anchor);
        p->slot[p->n_slots] |= 1u << d->decl_anchor;
    }
    p->n_slots += d->n_layer;
    /* no lane counters to keep with the walk: the ring and the anchor ARE the drafter's state at the bank's frontier
     * (a draft writes neither) */
    p->drafter_at = NULL;
    p->drafter_frontier = NULL;
}

void pulsar_block_drafter_describe(const pulsar_block_drafter *d, char *out, size_t cap) {
    char headw[64];
    if (d->head_kind == PULSAR_BLOCK_HEAD_SELECTOR) snprintf(headw, sizeof headw, "CandidateSelector top-%u rank %u",
                                                             d->topk, d->sel.rank);
    else snprintf(headw, sizeof headw, "LiLiCorr top-%u", d->topk);
    int n = snprintf(out, cap, "%s block drafter: %u layers %u/%u/%u, ff %u, block %u, window %u, conv %ux/%u, "
                               "head %u rows%s, %s chain head, taps",
                     d->kind, d->n_layer, d->n_head, d->n_kv, d->head_dim, d->ff, d->block, d->window, d->conv_taps,
                     d->conv_group, d->head_rows,
                     d->head_ids ? " (a draft vocabulary)" : "", headw);
    for (uint32_t j = 0; j < d->n_taps && n > 0 && (size_t)n < cap; j++)
        n += snprintf(out + n, cap - (size_t)n, " %u@L%u%s", d->tap_block[j], d->tap_layer[j],
                      d->tap_sub[j] == PULSAR_BLOCK_TAP_MIXER ? "m" : "f");
}

/* ---- the lane ---------------------------------------------------------------------------------------------------- */

namespace {

/* Rows one absorb pass carries (a prompt chunk is taken in passes of this many: every Linear's prompt arm is one
 * fixed-order kernel per shape, so a row's bytes do not depend on the pass) and the rows one draft pass carries
 * (PULSAR_DRIVER_HEADS_MAX: four banks of an 8-row block). */
constexpr uint32_t kAbsorbRows = 512u;
constexpr uint32_t kDraftRows = PULSAR_DRIVER_HEADS_MAX;

/* the absorb's region */
struct absorb_layout { uint64_t c, cn, stage, kb, vb, ws, ws_bytes, gqa, gqa_bytes, total; };
/* the draft's region */
struct draft_layout {
    uint64_t ids, h, x, cx, t, ob, coef, q, k, v, a, g, u, m, stage, ws, ws_bytes, gqa, gqa_bytes, logits;
    uint64_t desc, desc_stride;   /* every layer's attention descriptors (pulsar_gqa_plan), desc_stride apart */
    /* the chain head's inputs and outputs, [bank][slot]...: the candidates and their log-probs, the scores, the walk
     * (both kinds); LILICORR's pass rows, anchor rows and validity and its workspace (pulsar_cuda_lilicorr.h);
     * SELECTOR's projected rows and anchor ids (pulsar_cuda_candsel.h) */
    uint64_t cand_id, cand_lp, pass, anchor, valid, scores, chain_ids, chain_rows, head_ws, head_ws_bytes;
    uint64_t hp, anchor_ids, total;
};

uint64_t multi_ws(int in, const int *outs, int n, uint32_t rows) {
    pulsar_rows_linear l[4] = {};
    const pulsar_rows_linear *lp[4];
    for (int j = 0; j < n; j++) {
        l[j].in = in;
        l[j].out = outs[j];
        lp[j] = &l[j];
    }
    return a256(pulsar_rows_linear_multi_workspace_bytes(lp, n, (int)rows));
}

absorb_layout absorb_of(const pulsar_block_drafter *d) {
    const uint64_t R = kAbsorbRows, E = d->hidden, KV = (uint64_t)d->n_kv * d->head_dim;
    absorb_layout L{};
    uint64_t o = 0;
    L.c = o; o += a256(R * E * 2u);
    L.cn = o; o += a256(R * E * 2u);
    L.stage = o; o += a256(R * std::max<uint64_t>(E, 2u * KV) * 4u);
    L.kb = o; o += a256(R * KV * 2u);
    L.vb = o; o += a256(R * KV * 2u);
    const int fo[1] = {(int)E}, kvo[2] = {(int)KV, (int)KV};
    L.ws = o; L.ws_bytes = std::max(multi_ws((int)(d->n_taps * E), fo, 1, R), multi_ws((int)E, kvo, 2, R));
    o += L.ws_bytes;
    const pulsar_gqa_layer g = gqa_decl(d, NULL, NULL, false);
    L.gqa = o; L.gqa_bytes = a256(pulsar_gqa_store_workspace_bytes(&g, (uint32_t)R)); o += L.gqa_bytes;
    L.total = o;
    return L;
}

draft_layout draft_of(const pulsar_block_drafter *d) {
    const uint64_t R = kDraftRows, E = d->hidden, Q = (uint64_t)d->n_head * d->head_dim,
                   KV = (uint64_t)d->n_kv * d->head_dim, F = d->ff, CP = 2ull * d->conv_taps * (E / d->conv_group);
    draft_layout L{};
    uint64_t o = 0;
    L.ids = o; o += a256(R * 4u);
    L.h = o; o += a256(R * E * 2u);
    L.x = o; o += a256(R * E * 2u);
    L.cx = o; o += a256(R * E * 2u);
    L.t = o; o += a256(R * E * 2u);
    L.ob = o; o += a256(R * E * 2u);
    L.coef = o; o += a256(R * CP * 2u);
    L.q = o; o += a256(R * Q * 2u);
    L.k = o; o += a256(R * KV * 2u);
    L.v = o; o += a256(R * KV * 2u);
    L.a = o; o += a256(R * Q * 2u);
    L.g = o; o += a256(R * F * 2u);
    L.u = o; o += a256(R * F * 2u);
    L.m = o; o += a256(R * F * 2u);
    /* the f32 staging of the arms that do not emit bf16 themselves: the widest call (the head's vocabulary) */
    L.stage = o;
    o += a256(R * std::max<uint64_t>(std::max<uint64_t>(2u * F, Q + 2u * KV), std::max<uint64_t>(d->head_rows, CP)) * 4u);
    const int qkv[3] = {(int)Q, (int)KV, (int)KV}, gu[2] = {(int)F, (int)F}, eo[1] = {(int)E}, cp[1] = {(int)CP};
    L.ws = o;
    L.ws_bytes = std::max(std::max(multi_ws((int)E, qkv, 3, R), multi_ws((int)E, gu, 2, R)),
                          std::max(std::max(multi_ws((int)F, eo, 1, R), multi_ws((int)Q, eo, 1, R)),
                                   multi_ws((int)E, cp, 1, R)));
    if (d->head_kind == PULSAR_BLOCK_HEAD_SELECTOR) {   /* its head over a draft vocabulary, its projection */
        const int hr[1] = {(int)d->head_rows}, sr[1] = {(int)d->sel.rank};
        L.ws_bytes = std::max(L.ws_bytes, std::max(multi_ws((int)E, hr, 1, R), multi_ws((int)E, sr, 1, R)));
    }
    o += L.ws_bytes;
    const pulsar_gqa_layer g = gqa_decl(d, NULL, NULL, true);
    /* sized for any bound: the window's chunks cap a row's partials whatever the context */
    L.gqa = o; L.gqa_bytes = a256(pulsar_gqa_workspace_bytes(&g, (uint32_t)R, 2u * d->window + 1024u));
    o += L.gqa_bytes;
    L.desc_stride = a256(pulsar_gqa_desc_bytes((uint32_t)R));
    L.desc = o; o += (uint64_t)d->n_layer * L.desc_stride;
    L.logits = o; o += a256(R * d->head_rows * 2u);   /* bf16: the head's rows as its producer emits them */
    const uint64_t NB = R / d->block, SL = d->block - 1u;   /* the banks a pass carries, their slots */
    L.cand_id = o; o += a256(NB * SL * d->topk * 4u);
    L.cand_lp = o; o += a256(NB * SL * d->topk * 4u);
    L.scores = o; o += a256(NB * SL * d->topk * d->topk * 4u);
    L.chain_ids = o; o += a256(NB * SL * 4u);
    L.chain_rows = o; o += a256(NB * SL * 4u);
    if (d->head_kind == PULSAR_BLOCK_HEAD_LILICORR) {
        L.pass = o; o += a256(NB * SL * E * 2u);
        L.anchor = o; o += a256(NB * E * 2u);
        L.valid = o; o += a256(NB);
        L.head_ws = o; L.head_ws_bytes = a256(pulsar_lilicorr_head_ws_bytes((uint32_t)NB)); o += L.head_ws_bytes;
    } else {
        L.hp = o; o += a256(R * d->sel.rank * 2u);
        L.anchor_ids = o; o += a256(NB * 4u);
    }
    L.total = o;
    return L;
}

/* The candidate head's device weights: each Linear the rows front door's reference (decode rows: a bank's bytes never
 * depend on the banks beside it) and its bias, each vector its device pointer. */
bool head_weights(const pulsar_session *s, const pulsar_block_drafter *d, pulsar_lilicorr_head *w) {
    const pulsar_model *m = &s->engine->model;
    const pulsar_block_head &H = d->head_lc;
    bool ok = true;
    auto vec = [&](const pulsar_tensor *t, const char *what) {
        const uint16_t *p = (const uint16_t *)pulsar_weight_device_ptr(m, t, what);
        ok = ok && p;
        return p;
    };
    auto lin = [&](pulsar_lilicorr_linear *o, const pulsar_tensor *t, const pulsar_tensor *bias, int in, int out,
                   const char *what) {
        ok = ok && pulsar_linear_rows_ref(m, t, in, out, false, what, &o->l);
        o->b = vec(bias, what);
    };
    const int E = (int)d->hidden, LH = (int)PULSAR_LILICORR_HIDDEN;
    *w = pulsar_lilicorr_head{};
    lin(&w->token_proj, H.token_proj, H.token_proj_b, E, LH, "lilicorr token_proj");
    lin(&w->pass_hidden_proj, H.pass_proj, H.pass_proj_b, E, LH, "lilicorr pass_hidden_proj");
    lin(&w->context_proj, H.context_proj, H.context_proj_b, E, LH, "lilicorr context_proj");
    w->feature_norm_w = vec(H.feature_norm_w, "lilicorr feature_norm");
    w->feature_norm_b = vec(H.feature_norm_b, "lilicorr feature_norm bias");
    w->feature_mlp_up_w = vec(H.feature_up_w, "lilicorr feature_mlp up");
    w->feature_mlp_up_b = vec(H.feature_up_b, "lilicorr feature_mlp up bias");
    lin(&w->feature_mlp_down, H.feature_down, H.feature_down_b, LH, LH, "lilicorr feature_mlp down");
    w->slot_embedding = vec(H.slot_embedding, "lilicorr slot_embedding");
    w->rank_embedding = vec(H.rank_embedding, "lilicorr rank_embedding");
    w->relative_slot_bias = vec(H.relative_slot_bias, "lilicorr relative_slot_bias");
    w->same_slot_bias = vec(H.same_slot_bias, "lilicorr same_slot_bias");
    for (uint32_t l = 0; l < PULSAR_LILICORR_LAYERS; l++) {
        pulsar_lilicorr_layer &Y = w->layers[l];
        const auto &X = H.layer[l];
        Y.lat_attn_norm = vec(X.attn_norm, "lilicorr attn_norm");
        lin(&Y.lat_attn_in, X.attn_in, X.attn_in_b, LH, 3 * LH, "lilicorr attn in_proj");
        lin(&Y.lat_attn_out, X.attn_out, X.attn_out_b, LH, LH, "lilicorr attn out_proj");
        Y.lat_mlp_norm = vec(X.mlp_norm, "lilicorr mlp_norm");
        lin(&Y.lat_mlp_up, X.mlp_up, X.mlp_up_b, LH, 2 * LH, "lilicorr mlp up");
        lin(&Y.lat_mlp_down, X.mlp_down, X.mlp_down_b, 2 * LH, LH, "lilicorr mlp down");
    }
    w->output_norm = vec(H.output_norm, "lilicorr output_norm");
    w->anchor_norm = vec(H.anchor_norm, "lilicorr anchor_norm");
    lin(&w->factor_input_proj, H.factor_in, H.factor_in_b, 3 * LH, LH, "lilicorr factor_input_proj");
    lin(&w->out_head, H.out_head, H.out_head_b, LH, LH, "lilicorr out_head");
    lin(&w->in_head, H.in_head, H.in_head_b, LH, LH, "lilicorr in_head");
    lin(&w->anchor_out_head, H.anchor_out_head, H.anchor_out_head_b, LH, LH, "lilicorr anchor_out_head");
    w->vector_eps = H.vector_eps;
    w->logit_scale = H.logit_scale;
    if (!ok) fprintf(stderr, "pulsar: block drafter: the candidate head's weights did not resolve -- refusing the session\n");
    return ok;
}

}  // namespace

struct pulsar_block_lane {
    const pulsar_block_drafter *d;
    uint32_t n_banks, max_rows;
    uint64_t tap_width;            /* n_taps x hidden */
    pulsar_gpu_tensor *taps;       /* [max_rows][tap_width] bf16: the last trunk step's tapped rows */
    pulsar_gpu_tensor *arena;      /* the absorb's or the draft's region (never both at once) */
    absorb_layout al;
    draft_layout dl;
    /* host staging */
    std::vector<uint32_t> row_seq, row_pos, seq_bank;
    std::vector<pulsar_gqa_seq> seqs;
    /* the last draft pass's head outputs, read back: [bank][slot][...] */
    std::vector<int32_t> cand_id, chain_ids;
    std::vector<float> cand_lp, scores;
    pulsar_lilicorr_head head;   /* LILICORR: the candidate head's device weights (fixed for the lane's life) */
    /* the draft pass's host half (pinned: its uploads read them after the call returns) */
    int32_t *h_tok;
    int32_t *h_anchor;           /* SELECTOR: each bank's carry, slot 0's predecessor */
    uint8_t *h_valid;
    uint8_t *h_desc;
    pulsar_gqa_call call[PULSAR_BLOCK_MAX_LAYERS];
    bool announced;
};

pulsar_block_lane *pulsar_block_lane_create(pulsar_session *s, uint32_t n_banks, uint32_t max_rows) {
    const pulsar_block_drafter *d = DR(s->engine);
    if (!d) return NULL;
    pulsar_block_lane *l = new pulsar_block_lane();
    l->d = d;
    l->n_banks = n_banks;
    l->max_rows = max_rows;
    l->tap_width = (uint64_t)d->n_taps * d->hidden;
    l->al = absorb_of(d);
    l->dl = draft_of(d);
    l->taps = pulsar_gpu_tensor_alloc((uint64_t)max_rows * l->tap_width * 2u);
    l->arena = pulsar_gpu_tensor_alloc(std::max(l->al.total, l->dl.total));
    l->row_seq.resize(std::max(kAbsorbRows, kDraftRows));
    l->row_pos.resize(l->row_seq.size());
    l->seqs.resize(l->row_seq.size());
    l->seq_bank.resize(l->row_seq.size());
    const size_t nbs = (size_t)(kDraftRows / d->block) * (d->block - 1u);   /* bank slots a pass carries */
    l->cand_id.resize(nbs * d->topk);
    l->cand_lp.resize(nbs * d->topk);
    l->chain_ids.resize(nbs);
    l->scores.resize(nbs * d->topk * d->topk);
    l->h_tok = (int32_t *)pulsar_gpu_host_alloc((uint64_t)kDraftRows * 4u);
    l->h_anchor = (int32_t *)pulsar_gpu_host_alloc((uint64_t)kDraftRows * 4u);
    l->h_valid = (uint8_t *)pulsar_gpu_host_alloc(kDraftRows);
    l->h_desc = (uint8_t *)pulsar_gpu_host_alloc((uint64_t)d->n_layer * l->dl.desc_stride);
    if (!l->h_tok || !l->h_anchor || !l->h_valid || !l->h_desc ||
        (d->head_kind == PULSAR_BLOCK_HEAD_LILICORR && !head_weights(s, d, &l->head))) {
        pulsar_block_lane_free(l);
        return NULL;
    }
    if (!l->taps || !l->arena) {
        fprintf(stderr, "pulsar: block drafter: lane allocation failed (%u rows of %llu taps)\n", max_rows,
                (unsigned long long)l->tap_width);
        pulsar_block_lane_free(l);
        return NULL;
    }
    return l;
}

void pulsar_block_lane_free(pulsar_block_lane *l) {
    if (!l) return;
    pulsar_gpu_tensor_free(l->taps);
    pulsar_gpu_tensor_free(l->arena);
    pulsar_gpu_host_free(l->h_tok);
    pulsar_gpu_host_free(l->h_anchor);
    pulsar_gpu_host_free(l->h_valid);
    pulsar_gpu_host_free(l->h_desc);
    delete l;
}

uint16_t *pulsar_block_tap_dst(pulsar_driver_state *ds, uint32_t il, uint32_t sub, uint32_t n, uint32_t *j,
                               uint64_t *ld_dst, bool *refused) {
    pulsar_block_lane *l = ds ? ds->block : NULL;
    if (!l) return NULL;
    const pulsar_block_drafter *d = l->d;
    for (; *j < d->n_taps; (*j)++) {
        if (d->tap_layer[*j] != il || d->tap_sub[*j] != sub) continue;
        if (n > l->max_rows) {
            *refused = !fail("a trunk step wider than the tap rows");
            return NULL;
        }
        *ld_dst = l->tap_width;
        return (uint16_t *)dptr(l->taps) + (uint64_t)(*j)++ * d->hidden;
    }
    return NULL;
}

bool pulsar_block_tap(pulsar_driver_state *ds, uint32_t il, uint32_t sub, const uint16_t *h, uint64_t ld, uint32_t n) {
    const uint32_t E = ds && ds->block ? ds->block->d->hidden : 0u;
    return pulsar_block_tap_rows(ds, il, sub, n, [&](uint16_t *dst, uint64_t ld_dst) {
        return pulsar_blockdraft_rows_copy_launch(h, ld, n, E, dst, ld_dst, 0) == 0 || fail("the tap copy refused");
    });
}

namespace {

pulsar_block_lane *LANE(const pulsar_session *s) { return s->driver ? s->driver->block : NULL; }

/* ---- the reference grade's dump (diagnostic: PULSAR_BLOCK_DUMP=<dir>, read once) ------------------------------------
 * What tools/blockdraft/ref.py replays against the PyTorch reference of the drafter: every absorbed context row (its
 * bank, position and taps) to <dir>/ctx.bin, and the first PULSAR_BLOCK_DUMP_PASSES (default 8) draft passes (their
 * banks, carries, positions, each row's final hidden and top-k) to <dir>/draft.bin.  Little-endian records:
 *   ctx    u32 n, i32 pos[n], i32 bank[n], u32 width, bf16 taps[n][width]
 *   draft  u32 nb, u32 block, u32 hidden, u32 topk, i32 bank[nb], i32 carry[nb], i32 P[nb],
 *          bf16 o[nb block][hidden], i32 cand[nb][block - 1][topk], f32 lp[nb][block - 1][topk],
 *          f32 scores[nb][block - 1][topk][topk], i32 chain[nb][block - 1] */
const char *dump_dir() {
    static const char *dir = NULL;
    static bool read = false;
    if (!read) {
        read = true;
        const char *v = getenv("PULSAR_BLOCK_DUMP");
        dir = v && v[0] ? v : NULL;
        if (dir) fprintf(stderr, "pulsar: block drafter: dumping context rows and draft passes to %s\n", dir);
    }
    return dir;
}
FILE *dump_open(const char *name) {
    char p[1024];
    snprintf(p, sizeof p, "%s/%s", dump_dir(), name);
    return fopen(p, "ab");
}

void dump_ctx(pulsar_block_lane *l, uint32_t row0, const int32_t *pos, const int32_t *bank, uint32_t n) {
    if (!dump_dir() || n == 0) return;
    std::vector<uint16_t> taps((size_t)n * l->tap_width);
    if (!pulsar_gpu_tensor_read(l->taps, (uint64_t)row0 * l->tap_width * 2u, taps.data(), taps.size() * 2u)) return;
    FILE *f = dump_open("ctx.bin");
    if (!f) return;
    const uint32_t w = (uint32_t)l->tap_width;
    fwrite(&n, 4, 1, f);
    fwrite(pos, 4, n, f);
    fwrite(bank, 4, n, f);
    fwrite(&w, 4, 1, f);
    fwrite(taps.data(), 2, taps.size(), f);
    fclose(f);
}

const void *wp(const pulsar_session *s, const pulsar_tensor *t, const char *what) {
    return pulsar_weight_device_ptr(&s->engine->model, t, what);
}

/* n Linears of one bf16 input -> their bf16 rows (the producer's one rounding; the nvfp4 arm's global scale in its
 * epilogue), the front door's arm by each weight's format and the row kind */
bool linears(pulsar_session *s, int n, const pulsar_tensor *const *t, int in, const int *out, const char *what,
             const uint16_t *x, uint32_t rows, bool prompt, float *stage, uint16_t *const *yb, void *ws,
             uint64_t ws_bytes) {
    pulsar_rows_linear lr[4];
    const pulsar_rows_linear *lp[4];
    float *st[4];
    uint64_t off = 0;
    for (int j = 0; j < n; j++) {
        if (!pulsar_linear_rows_ref(&s->engine->model, t[j], in, out[j], prompt, what, &lr[j])) return fail(what);
        lp[j] = &lr[j];
        st[j] = stage + off;
        off += (uint64_t)rows * (uint64_t)out[j];
    }
    return pulsar_rows_linear_multi_bf16_launch(lp, n, x, (int)rows, st, yb, ws, ws_bytes, 0) == 0 || fail(what);
}

/* bank b's ring of drafter layer l, as a GQA sequence */
pulsar_gqa_seq ring_of(const pulsar_session *s, const pulsar_block_drafter *d, uint32_t l, uint32_t b) {
    const pulsar_state_set *ss = s->states;
    pulsar_gpu_tensor *t = pulsar_state_tensor(ss, ss->plan->n_trunk + l, d->decl_kv);
    const pulsar_state_decl &dk = ss->plan->decl[d->decl_kv];
    const uint64_t bb = pulsar_state_decl_bank_bytes(&dk);
    pulsar_gqa_seq q{};
    if (!t) return q;
    q.kv = (uint8_t *)dptr(t) + (uint64_t)b * bb;
    q.bytes = bb;
    q.cap = (uint32_t)(bb / dk.bytes);
    return q;
}

/* The context absorb: rows [row0, row0 + n) of the last trunk step (their taps), at positions pos[i] of banks bank[i]
 * -- each bank's rows one run of consecutive positions -- into every drafter layer's ring, and each bank's last row's
 * cn into its anchor.  `prompt`: the prompt arms (a prompt chunk's rows), else the decode arms. */
bool absorb_rows(pulsar_session *s, uint32_t row0, const int32_t *pos, const int32_t *bank, uint32_t n, bool prompt) {
    pulsar_block_lane *l = LANE(s);
    const pulsar_block_drafter *d = l->d;
    const pulsar_state_set *ss = s->states;
    const uint32_t E = d->hidden, KV = d->n_kv * d->head_dim;
    if (row0 + n > l->max_rows) return fail("absorbed rows past the tap rows");
    /* L307, armed only: the lane's arena filled first (the absorb's region, never live across a call) */
    if (!pulsar_gpu_scratch_poison_step(&l->arena, 1)) return false;
    uint8_t *base = (uint8_t *)dptr(l->arena);
    const absorb_layout &A = l->al;
    uint16_t *c = (uint16_t *)(base + A.c), *cn = (uint16_t *)(base + A.cn), *kb = (uint16_t *)(base + A.kb),
             *vb = (uint16_t *)(base + A.vb);
    float *stage = (float *)(base + A.stage);
    const uint16_t *hn = (const uint16_t *)wp(s, d->hidden_norm, "drafter hidden_norm");
    pulsar_gpu_tensor *anchor =
        d->decl_anchor != UINT32_MAX ? pulsar_state_tensor(ss, ss->plan->n_trunk, d->decl_anchor) : NULL;
    if (!hn || (d->decl_anchor != UINT32_MAX && !anchor)) return fail("the drafter's context norm or anchor state is missing");
    dump_ctx(l, row0, pos, bank, n);
    for (uint32_t p0 = 0; p0 < n; p0 += kAbsorbRows) {
        const uint32_t m = std::min(kAbsorbRows, n - p0);
        const uint16_t *x = (const uint16_t *)dptr(l->taps) + (uint64_t)(row0 + p0) * l->tap_width;
        {
            const pulsar_tensor *t[1] = {d->fc};
            const int out[1] = {(int)E};
            uint16_t *yb[1] = {c};
            if (!linears(s, 1, t, (int)l->tap_width, out, "drafter fc", x, m, prompt, stage, yb, base + A.ws, A.ws_bytes))
                return false;
        }
        if (pulsar_rows_rmsnorm_launch(c, E, m, E, hn, d->eps, cn, E, 0) != 0) return fail("the context norm refused");
        /* the rows' sequences: one a run of a bank (the GQA plan refuses a bank in two runs) */
        uint32_t ns = 0;
        for (uint32_t i = 0; i < m; i++) {
            const uint32_t r = p0 + i;
            if (i == 0 || bank[r] != bank[r - 1]) {
                if ((uint32_t)bank[r] >= ss->n_banks) return fail("an absorbed row's bank is outside the pool");
                l->seq_bank[ns++] = (uint32_t)bank[r];
            }
            l->row_seq[i] = ns - 1u;
            l->row_pos[i] = (uint32_t)pos[r];
        }
        for (uint32_t li = 0; li < d->n_layer; li++) {
            const pulsar_block_layer &L = d->layer[li];
            const pulsar_tensor *t[2] = {L.k, L.v};
            const int out[2] = {(int)KV, (int)KV};
            uint16_t *yb[2] = {kb, vb};
            if (!linears(s, 2, t, (int)E, out, "drafter k / v", cn, m, prompt, stage, yb, base + A.ws, A.ws_bytes))
                return false;
            for (uint32_t k = 0; k < ns; k++) l->seqs[k] = ring_of(s, d, li, l->seq_bank[k]);
            const pulsar_gqa_layer G =
                gqa_decl(d, (const uint16_t *)wp(s, L.q_norm, "drafter q_norm"), (const uint16_t *)wp(s, L.k_norm, "drafter k_norm"), false);
            pulsar_gqa_io io{};
            io.k = {kb, KV, d->head_dim};
            io.v = {vb, KV, d->head_dim};
            io.workspace = base + A.gqa;
            io.workspace_bytes = A.gqa_bytes;
            if (!G.q_norm_w || !G.k_norm_w ||
                pulsar_gqa_store(&G, l->seqs.data(), ns, l->row_seq.data(), l->row_pos.data(), m, &io, 0) != 0)
                return fail("the context store refused");
        }
        /* LILICORR: each bank's last row of this pass is its anchor (a later pass's overwrites it) */
        for (uint32_t i = 0; anchor && i < m; i++) {
            const uint32_t r = p0 + i;
            if (i + 1u < m && bank[r + 1] == bank[r]) continue;
            if (!pulsar_gpu_tensor_copy_async(anchor, (uint64_t)bank[r] * E * 2u, l->arena, A.cn + (uint64_t)i * E * 2u,
                                              (uint64_t)E * 2u))
                return fail("the anchor copy failed");
        }
    }
    return true;
}

/* ---- the draft ------------------------------------------------------------------------------------------------------
 * One pass over nb <= kDraftRows / block banks: the block forward, the drafter's head over every row, each bank's slot
 * rows' top-k, the candidate head and its walk; the candidates, the scores and the chain read back.  Bank i's rows
 * are [i block, (i + 1) block) at positions P_i ...
 *
 * In two halves, as the driver's steps are (L296 perf): the HOST half stages everything the launches read -- the
 * rows' ids, every layer's attention descriptors (the banks' rings, bounded at a power of two above the rows'
 * positions), each bank's anchor row and validity -- and the LAUNCH half reads only the device and the pass's shape,
 * so it runs as ONE captured CUDA graph per (lane, banks, bound) (the L119 segment cache: captured on a key's warm use,
 * replayed after; eager while the reference dump reads the pass).  pos[i] is bank i's position P_i: its frontier, or
 * (L306) the warmup sweep's stand-in for a deeper one -- a pass writes nothing but the lane's arena. */
bool draft_pass(pulsar_session *s, const uint32_t *banks, const uint32_t *pos, const int32_t *carry, uint32_t nb) {
    pulsar_block_lane *l = LANE(s);
    const pulsar_block_drafter *d = l->d;
    const pulsar_state_set *ss = s->states;
    const uint32_t B = d->block, R = nb * B, E = d->hidden, Q = d->n_head * d->head_dim, KV = d->n_kv * d->head_dim,
                   F = d->ff, CP = 2u * d->conv_taps * (E / d->conv_group), V = d->vocab, HR = d->head_rows,
                   half = d->conv_taps * (E / d->conv_group), SL = B - 1u, K = d->topk;
    const bool lili = d->head_kind == PULSAR_BLOCK_HEAD_LILICORR;
    if (R > kDraftRows || nb == 0) return fail("a draft pass outside the lane's rows");
    /* L307, armed only: the lane's arena (the absorb's and the draft's, never live across a pass) filled first */
    if (!pulsar_gpu_scratch_poison_step(&l->arena, 1)) return false;
    uint8_t *base = (uint8_t *)dptr(l->arena);
    const draft_layout &D = l->dl;
    /* ---- the host half */
    uint32_t last = 0;
    for (uint32_t i = 0; i < nb; i++) {
        const uint32_t P = pos[i];
        if (carry[i] < 0 || (uint32_t)carry[i] >= V) return fail("a carry outside the vocabulary");
        l->h_tok[i * B] = carry[i];
        l->h_anchor[i] = carry[i];
        for (uint32_t j = 1; j < B; j++) l->h_tok[i * B + j] = d->mask_id;
        for (uint32_t j = 0; j < B; j++) {
            l->row_seq[i * B + j] = i;
            l->row_pos[i * B + j] = P + j;
        }
        l->h_valid[i] = 1;   /* a drafted bank holds a committed row (P > 0): its anchor is real */
        last = std::max(last, P + B - 1u);
    }
    uint32_t bound = 256u;   /* the attention's bound: a graph is recaptured once per doubling of the context */
    while (bound <= last) bound *= 2u;
    bool ok = pulsar_gpu_tensor_write_async(l->arena, D.ids, l->h_tok, (uint64_t)R * 4u) != 0;
    if (lili) {   /* the candidate head's anchor rows (cn of each bank's last committed row) and their validity */
        pulsar_gpu_tensor *anchor = pulsar_state_tensor(ss, ss->plan->n_trunk, d->decl_anchor);
        ok = ok && anchor && pulsar_gpu_tensor_write_async(l->arena, D.valid, l->h_valid, nb) != 0;
        for (uint32_t i = 0; ok && i < nb; i++)
            ok = pulsar_gpu_tensor_copy_async(l->arena, D.anchor + (uint64_t)i * E * 2u, anchor,
                                              (uint64_t)banks[i] * E * 2u, (uint64_t)E * 2u) != 0;
    } else {      /* the selector's anchors: the carries */
        ok = ok && pulsar_gpu_tensor_write_async(l->arena, D.anchor_ids, l->h_anchor, (uint64_t)nb * 4u) != 0;
    }
    std::vector<pulsar_gqa_layer> G(d->n_layer);
    for (uint32_t li = 0; ok && li < d->n_layer; li++) {
        const pulsar_block_layer &L = d->layer[li];
        G[li] = gqa_decl(d, (const uint16_t *)wp(s, L.q_norm, "drafter q_norm"),
                         (const uint16_t *)wp(s, L.k_norm, "drafter k_norm"), true);
        for (uint32_t i = 0; i < nb; i++) l->seqs[i] = ring_of(s, d, li, banks[i]);
        ok = G[li].q_norm_w && G[li].k_norm_w &&
             pulsar_gqa_plan(&G[li], l->seqs.data(), nb, l->row_seq.data(), l->row_pos.data(), R, bound,
                             l->h_desc + (uint64_t)li * D.desc_stride, D.desc_stride, &l->call[li]) == 0;
    }
    ok = ok && pulsar_gpu_tensor_write_async(l->arena, D.desc, l->h_desc, (uint64_t)d->n_layer * D.desc_stride) != 0;
    if (!ok) return fail("the draft pass's host half refused");
    /* ---- the launch half */
    uint16_t *h = (uint16_t *)(base + D.h), *x = (uint16_t *)(base + D.x), *cx = (uint16_t *)(base + D.cx),
             *t = (uint16_t *)(base + D.t), *ob = (uint16_t *)(base + D.ob), *coef = (uint16_t *)(base + D.coef),
             *q = (uint16_t *)(base + D.q), *k = (uint16_t *)(base + D.k), *v = (uint16_t *)(base + D.v),
             *a = (uint16_t *)(base + D.a), *g = (uint16_t *)(base + D.g), *u = (uint16_t *)(base + D.u),
             *mm = (uint16_t *)(base + D.m), *logits = (uint16_t *)(base + D.logits);
    float *stage = (float *)(base + D.stage);
    void *ws = base + D.ws;
    const uint16_t *table = (const uint16_t *)wp(s, d->embed, "drafter embed (the target's)");
    const uint16_t *on = (const uint16_t *)wp(s, d->out_norm, "drafter norm");
    const int32_t *id_map = d->head_ids ? (const int32_t *)wp(s, d->head_ids, "drafter head ids") : NULL;
    const uint16_t *pred = lili ? NULL : (const uint16_t *)wp(s, d->sel.pred, "selector predecessor codebook");
    const uint16_t *succ = lili ? NULL : (const uint16_t *)wp(s, d->sel.succ, "selector successor codebook");
    if (!table || !on || (d->head_ids && !id_map) || (!lili && (!pred || !succ)))
        return fail("the drafter's embedding, final norm, head ids or selector codebooks are missing");
    auto launches = [&]() -> bool {
        pulsar_head_decl hd{};
        hd.n_embd = E;
        hd.n_logits = V;
        hd.n_real = V;
        hd.logit_scale = 1.0f;
        if (pulsar_head_embed_launch(&hd, table, V, (const int32_t *)(base + D.ids), R, h, E, 0) != 0)
            return fail("the block's embedding refused");
        for (uint32_t li = 0; li < d->n_layer; li++) {
            const pulsar_block_layer &L = d->layer[li];
            const uint16_t *in_n = (const uint16_t *)wp(s, L.in_norm, "drafter in_norm");
            const uint16_t *post_n = (const uint16_t *)wp(s, L.post_norm, "drafter post_norm");
            const uint16_t *ab = (const uint16_t *)wp(s, L.ac_base, "drafter attn conv base");
            const uint16_t *mb = (const uint16_t *)wp(s, L.mc_base, "drafter mlp conv base");
            if (!in_n || !post_n || !ab || !mb) return false;
            /* x = rmsnorm(h) (layer 0), else h += the previous layer's output and x its norm */
            if ((li == 0 ? pulsar_rows_rmsnorm_launch(h, E, R, E, in_n, d->eps, x, E, 0)
                         : pulsar_rows_add_launch(h, E, t, E, R, E, in_n, d->eps, x, E, 0)) != 0)
                return fail("the input norm refused");
            /* the attention conv's taps, its prepare; q / k / v; the block attention; o; the conv's finish (side 1) */
            {
                const pulsar_tensor *w[1] = {L.ac_proj};
                const int out[1] = {(int)CP};
                uint16_t *yb[1] = {coef};
                if (!linears(s, 1, w, (int)E, out, "drafter attn conv projection", x, R, false, stage, yb, ws, D.ws_bytes))
                    return false;
            }
            if (pulsar_blockdraft_conv_launch(x, E, coef, CP, ab, d->conv_taps, d->conv_group, B, R, E, cx, E, 0) != 0)
                return fail("the attention conv refused");
            {
                const pulsar_tensor *w[3] = {L.q, L.k, L.v};
                const int out[3] = {(int)Q, (int)KV, (int)KV};
                uint16_t *yb[3] = {q, k, v};
                if (!linears(s, 3, w, (int)E, out, "drafter q / k / v", cx, R, false, stage, yb, ws, D.ws_bytes))
                    return false;
            }
            pulsar_gqa_io io{};
            io.q = {q, Q, d->head_dim};
            io.k = {k, KV, d->head_dim};
            io.v = {v, KV, d->head_dim};
            io.out_bf16 = a;
            io.ld_out = Q;
            io.workspace = base + D.gqa;
            io.workspace_bytes = D.gqa_bytes;
            if (pulsar_gqa_launch(&G[li], &l->call[li], base + D.desc + (uint64_t)li * D.desc_stride, &io, 0) != 0)
                return fail("the block attention refused");
            {
                const pulsar_tensor *w[1] = {L.o};
                const int out[1] = {(int)E};
                uint16_t *yb[1] = {ob};
                if (!linears(s, 1, w, (int)Q, out, "drafter o_proj", a, R, false, stage, yb, ws, D.ws_bytes)) return false;
            }
            if (pulsar_blockdraft_conv_launch(ob, E, coef + half, CP, ab + (uint64_t)d->conv_taps * E, d->conv_taps,
                                              d->conv_group, B, R, E, t, E, 0) != 0 ||
                pulsar_rows_add_launch(h, E, t, E, R, E, post_n, d->eps, x, E, 0) != 0)
                return fail("the attention's finish refused");
            /* the MLP conv's taps, its prepare; gate / up; SwiGLU; down; the conv's finish */
            {
                const pulsar_tensor *w[1] = {L.mc_proj};
                const int out[1] = {(int)CP};
                uint16_t *yb[1] = {coef};
                if (!linears(s, 1, w, (int)E, out, "drafter mlp conv projection", x, R, false, stage, yb, ws, D.ws_bytes))
                    return false;
            }
            if (pulsar_blockdraft_conv_launch(x, E, coef, CP, mb, d->conv_taps, d->conv_group, B, R, E, cx, E, 0) != 0)
                return fail("the MLP conv refused");
            {
                const pulsar_tensor *w[2] = {L.gate, L.up};
                const int out[2] = {(int)F, (int)F};
                uint16_t *yb[2] = {g, u};
                if (!linears(s, 2, w, (int)E, out, "drafter gate / up", cx, R, false, stage, yb, ws, D.ws_bytes))
                    return false;
            }
            if (pulsar_blockdraft_swiglu_launch(g, F, u, F, R, F, mm, F, 0) != 0) return fail("the SwiGLU refused");
            {
                const pulsar_tensor *w[1] = {L.down};
                const int out[1] = {(int)E};
                uint16_t *yb[1] = {ob};
                if (!linears(s, 1, w, (int)F, out, "drafter down", mm, R, false, stage, yb, ws, D.ws_bytes)) return false;
            }
            if (pulsar_blockdraft_conv_launch(ob, E, coef + half, CP, mb + (uint64_t)d->conv_taps * E, d->conv_taps,
                                              d->conv_group, B, R, E, t, E, 0) != 0)
                return fail("the MLP's finish refused");
        }
        /* o = rmsnorm(h + t) * norm; the drafter's head over every row */
        if (pulsar_rows_add_launch(h, E, t, E, R, E, on, d->eps, x, E, 0) != 0) return fail("the final norm refused");
        {
            const pulsar_tensor *w[1] = {d->head};
            const int out[1] = {(int)HR};
            uint16_t *yb[1] = {logits};
            if (!linears(s, 1, w, (int)E, out, "drafter lm_head", x, R, false, stage, yb, ws, D.ws_bytes)) return false;
        }
        /* each bank's slot rows' top-k candidates (vocabulary ids through the head's id map) and log-probs */
        for (uint32_t i = 0; i < nb; i++)
            if (pulsar_rows_topk_logprob_launch(logits + (uint64_t)(i * B + 1u) * HR, HR, SL, HR, K, id_map,
                                                (int32_t *)(base + D.cand_id) + (uint64_t)i * SL * K,
                                                (float *)(base + D.cand_lp) + (uint64_t)i * SL * K, 0) != 0 ||
                (lili && pulsar_blockdraft_rows_copy_launch(x + (uint64_t)(i * B + 1u) * E, E, SL, E,
                                                            (uint16_t *)(base + D.pass) + (uint64_t)i * SL * E, E,
                                                            0) != 0))
                return fail("the candidates' top-k refused");
        if (lili) return true;
        /* the selector (pulsar_cuda_candsel.h), in the graph: hidden_projection over every row, then the scores and
         * the walk from each bank's carry over its slot rows 1 .. block - 1 */
        {
            const pulsar_tensor *w[1] = {d->sel.proj};
            const int out[1] = {(int)d->sel.rank};
            uint16_t *yb[1] = {(uint16_t *)(base + D.hp)};
            if (!linears(s, 1, w, (int)E, out, "selector hidden_projection", x, R, false, stage, yb, ws, D.ws_bytes))
                return false;
        }
        return pulsar_candsel_launch(nb, SL, K, d->sel.rank, V, (const int32_t *)(base + D.cand_id),
                                     (const float *)(base + D.cand_lp),
                                     (const uint16_t *)(base + D.hp) + d->sel.rank, B, d->sel.rank,
                                     (const int32_t *)(base + D.anchor_ids), pred, succ, (float *)(base + D.scores),
                                     (int32_t *)(base + D.chain_ids), (int32_t *)(base + D.chain_rows), 0) == 0 ||
               fail("the candidate selector refused");
    };
    /* one graph per (lane, banks, bound); eager while the dump reads the pass */
    uint64_t key = 0;
    if (!dump_dir()) {
        key = 0x424c4b44ull;   /* "BLKD" */
        for (uint64_t v2 : {(uint64_t)(uintptr_t)l, (uint64_t)nb, (uint64_t)bound}) {
            key ^= v2 + 0x9e3779b97f4a7c15ull + (key << 6) + (key >> 2);
            key = (key ^ (key >> 30)) * 0xbf58476d1ce4e5b9ull;
        }
        key |= 1u;
    }
    const int seg = key ? pulsar_gpu_seg_enter(key) : 0;
    if (seg < 0) {
        ok = false;   /* L306: this shape's graph failed (the runtime's line names it): the pass refuses */
    } else if (seg == 1) {
        /* L306: a failed capture ran nothing and refuses -- no eager re-run */
        const bool body = launches();
        ok = pulsar_gpu_seg_exit(key, body ? 1 : 0) != 0;
    } else if (seg == 0) {
        ok = launches();
    }
    /* LILICORR: the candidate head after the graph, its own launches: its lattice attention's cuBLASLt batched GEMM
     * refuses inside a stream capture (status 14), so it runs eager behind the replay */
    ok = ok && (!lili || pulsar_lilicorr_head_launch(&l->head, nb, (const int32_t *)(base + D.cand_id),
                                            (const float *)(base + D.cand_lp), table, (const uint16_t *)(base + D.pass),
                                            (const uint16_t *)(base + D.anchor), (const uint8_t *)(base + D.valid),
                                            (float *)(base + D.scores), (int32_t *)(base + D.chain_ids),
                                            (int32_t *)(base + D.chain_rows), base + D.head_ws, D.head_ws_bytes,
                                            0) == 0 ||
                fail("the candidate head refused"));
    if (ok && !l->announced) {
        l->announced = true;
        fprintf(stderr, "pulsar: block drafter: the draft pass (%u bank(s) of %u rows) runs %s, %s; its chain is the "
                        "head's walk\n", nb, B, key ? "as one captured graph a (banks, bound)" : "eager (dump)",
                lili ? "the LiLiCorr candidate head behind it eager" : "the CandidateSelector inside it");
    }
    if (!ok) return false;
    if (!pulsar_gpu_tensor_read(l->arena, D.cand_id, l->cand_id.data(), (uint64_t)nb * SL * K * 4u) ||
        !pulsar_gpu_tensor_read(l->arena, D.cand_lp, l->cand_lp.data(), (uint64_t)nb * SL * K * 4u) ||
        !pulsar_gpu_tensor_read(l->arena, D.scores, l->scores.data(), (uint64_t)nb * SL * K * K * 4u) ||
        !pulsar_gpu_tensor_read(l->arena, D.chain_ids, l->chain_ids.data(), (uint64_t)nb * SL * 4u))
        return fail("could not read the head's chain");
    if (dump_dir()) {
        static int passes = -1;
        if (passes < 0) {
            const char *pv = getenv("PULSAR_BLOCK_DUMP_PASSES");
            passes = pv && pv[0] ? atoi(pv) : 8;
        }
        FILE *f = passes > 0 ? dump_open("draft.bin") : NULL;
        std::vector<uint16_t> o((size_t)R * E);
        if (f && pulsar_gpu_tensor_read(l->arena, D.x, o.data(), o.size() * 2u)) {
            passes--;
            int32_t bk[kDraftRows], P[kDraftRows];
            for (uint32_t i = 0; i < nb; i++) {
                bk[i] = (int32_t)banks[i];
                P[i] = (int32_t)pos[i];
            }
            const uint32_t hdr[4] = {nb, B, E, K};
            fwrite(hdr, 4, 4, f);
            fwrite(bk, 4, nb, f);
            fwrite(carry, 4, nb, f);
            fwrite(P, 4, nb, f);
            fwrite(o.data(), 2, o.size(), f);
            fwrite(l->cand_id.data(), 4, (size_t)nb * SL * K, f);
            fwrite(l->cand_lp.data(), 4, (size_t)nb * SL * K, f);
            fwrite(l->scores.data(), 4, (size_t)nb * SL * K * K, f);
            fwrite(l->chain_ids.data(), 4, (size_t)nb * SL, f);
        }
        if (f) fclose(f);
    }
    return true;
}

/* The chains of n banks, kDraftRows / block banks a pass: record i's carry at its bank's next position, its drafts
 * into the record -- slot j's top-1 (greedy) or a draw from slot j's row under the request's params at the draft's
 * key (sampled).  A copy chain's record (pulsar_spec_draft_req_init) is left as it is.  Each bank's rows are a block
 * of their own (one sequence, the decode arms: a bank's draft does not depend on the banks beside it). */
bool draft_chains(pulsar_session *s, const uint32_t *banks, uint64_t *const *rngs, spec_redraft_req *const *reqs,
                  uint32_t n, char *err, size_t errlen) {
    pulsar_block_lane *l = LANE(s);
    if (!l) {
        snprintf(err, errlen, "speculative decoding needs the block drafter's lane");
        return false;
    }
    const pulsar_block_drafter *d = l->d;
    const pulsar_state_set *ss = s->states;
    const uint32_t B = d->block, per = kDraftRows / B, V = d->vocab;
    uint32_t pb[kDraftRows], pi[kDraftRows], pp[kDraftRows];
    int32_t pc[kDraftRows];
    std::vector<spec_redraft_req *> chains;   /* the records this call drafted (the stop rule and the pick below) */
    for (uint32_t i0 = 0; i0 < n;) {
        uint32_t nb = 0, i = i0;
        for (; i < n && nb < per; i++) {
            spec_redraft_req *q = reqs[i];
            if (q->copied) continue;
            q->keep = 0;
            q->have_conf = true;
            q->refined[0] = q->next_base;
            const uint32_t P = ss->bank_pos[banks[i]];
            if (P == 0 || q->n_draft == 0 || (uint64_t)P + 1u >= ss->ctx) continue;   /* no context, or no room */
            if (q->temperature > 0.0f && !rngs[i]) {
                snprintf(err, errlen, "sampled speculation needs the request's rng");
                return false;
            }
            pb[nb] = banks[i];
            pp[nb] = P;
            pc[nb] = q->next_base;
            pi[nb++] = i;
        }
        i0 = i;
        if (nb == 0) continue;
        const double t0 = now_sec();
        if (!draft_pass(s, pb, pp, pc, nb)) {
            snprintf(err, errlen, "the block draft failed (see the log)");
            return false;
        }
        /* diagnostic (PULSAR_BLOCK_TIMING set, read once): the draft pass's wall, readback included, every 64 passes */
        static int timing = -1;
        if (timing < 0) timing = getenv("PULSAR_BLOCK_TIMING") ? 1 : 0;
        if (timing) {
            static double sum = 0;
            static uint32_t cnt = 0;
            sum += now_sec() - t0;
            if (++cnt == 64) {
                fprintf(stderr, "pulsar: block drafter: draft pass %.3f ms mean over 64 (%u bank(s) last)\n",
                        sum / cnt * 1e3, nb);
                sum = 0;
                cnt = 0;
            }
        }
        /* the sampled chains' draws, every bank's every slot in one launch of THE sampler (L302): slot j's row is
         * its candidates scored from the slot before's pick, so the walk runs on the device and one readback brings
         * the drafts */
        const uint32_t SL = B - 1u, TK = d->topk;
        int32_t drawn[PULSAR_GPU_CHAINS_MAX * PULSAR_GPU_CHAIN_SLOTS_MAX];   /* [chain][SL] */
        uint32_t chain_of[kDraftRows];
        {
            pulsar_gpu_chain_req cr[PULSAR_GPU_CHAINS_MAX];
            uint32_t nc = 0;
            for (uint32_t b = 0; b < nb; b++) {
                spec_redraft_req *q = reqs[pi[b]];
                chain_of[b] = UINT32_MAX;
                if (q->temperature <= 0.0f) continue;
                const uint32_t P = ss->bank_pos[pb[b]];
                uint32_t K = std::min(q->n_draft, B - 1u);
                if ((uint64_t)P + 1u + K > ss->ctx) K = (uint32_t)(ss->ctx - P - 1u);
                if (nc == PULSAR_GPU_CHAINS_MAX || SL > PULSAR_GPU_CHAIN_SLOTS_MAX) {
                    snprintf(err, errlen, "more sampled chains than a launch draws");
                    return false;
                }
                pulsar_gpu_chain_req &c = cr[nc];
                memset(&c, 0, sizeof(c));
                c.temperature = q->temperature;
                c.top_k = q->top_k;
                c.top_p = q->top_p;
                c.min_p = q->min_p;
                c.chain = b;
                c.n = K;
                for (uint32_t j = 1; j <= K; j++) c.key[j - 1] = pulsar_sample_key(*rngs[pi[b]], j);
                chain_of[b] = nc++;
            }
            if (nc > 0 && !s->draw_chains(l->arena, l->dl.cand_id, l->arena, l->dl.scores, SL, TK, V, cr, nc, drawn)) {
                snprintf(err, errlen, "the sampled chains' draws failed");
                return false;
            }
        }
        for (uint32_t b = 0; b < nb; b++) {
            spec_redraft_req *q = reqs[pi[b]];
            const uint32_t P = ss->bank_pos[pb[b]];
            uint32_t K = std::min(q->n_draft, B - 1u);
            if ((uint64_t)P + 1u + K > ss->ctx) K = (uint32_t)(ss->ctx - P - 1u);
            /* the walk over the head's scores: slot s's row is scores[s][prev] (slot 0: the start row) -- greedy the
             * device's walk (its first argmax), sampled a draw from the row's candidates under the request's params at
             * the draft's key (s + 1 places past the carry), as vLLM's candidate sampler draws from them */
            const int32_t *cand = &l->cand_id[(size_t)b * SL * TK];
            const float *sc = &l->scores[(size_t)b * SL * TK * TK];
            uint32_t prev = 0;
            for (uint32_t j = 1; j <= K; j++) {
                const float *rw = sc + ((size_t)(j - 1) * TK + prev) * TK;
                float mx = rw[0], sum = 0.f;
                for (uint32_t c = 1; c < TK; c++) mx = std::max(mx, rw[c]);
                for (uint32_t c = 0; c < TK; c++) sum += expf(rw[c] - mx);
                uint32_t pick = 0;
                if (q->temperature <= 0.0f) {
                    q->refined[j] = l->chain_ids[(size_t)b * SL + j - 1];
                    while (pick < TK && cand[(j - 1) * TK + pick] != q->refined[j]) pick++;
                } else {
                    const int dt = drawn[(size_t)chain_of[b] * SL + j - 1];   /* THE sampler's draw over the slot's candidates */
                    if (dt < 0) {
                        snprintf(err, errlen, "a draft slot's candidates have no distribution");
                        return false;
                    }
                    q->refined[j] = dt;
                    while (pick < TK && cand[(j - 1) * TK + pick] != dt) pick++;
                }
                if (pick == TK) {
                    snprintf(err, errlen, "a draft is not among its slot's candidates");
                    return false;
                }
                /* its probability among the slot's candidates: the head's belief in draft j given the walk so far
                 * (greedy: the row's max, known before the pick; sampled: the drawn candidate's q -- it moves only
                 * WHICH drafts are verified, never a token: the verify draws every token at its own key).  Calibrated
                 * on vLLM's goldens (480 verified drafts): by decile conf 0.35 / 0.53 / 0.69 / 0.88 / 0.98 accept
                 * 0.23 / 0.56 / 0.69 / 0.85 / 0.96 -- no rescale */
                q->conf[j - 1] = expf(rw[pick] - mx) / sum;
                prev = pick;
            }
            q->keep = K;
            chains.push_back(q);
        }
    }
    /* the shared draft schedule (spec_ops.h): each chain ends at its first draft under tau (that draft still verified),
     * then the verify pick (L301, verify_curve.cpp): the round's drafts by survival (the product of the confidences up
     * to each, a copy chain's beside the drafter's), as many as raise its tokens per ms on the measured curve, the
     * n banks' base rows beside them.  The block's pass is paid whatever the depth, so no drafter step enters: the
     * pick alone cuts a chain. */
    const float tau = pulsar_spec_tau(s->engine);
    for (spec_redraft_req *q : chains) q->keep = pulsar_spec_conf_keep(q->conf, q->keep, tau);
    static int unpriced = -1;   /* diagnostic (PULSAR_BLOCK_UNPRICED set, read once): the sweep's rule-off arm */
    if (unpriced < 0) unpriced = getenv("PULSAR_BLOCK_UNPRICED") ? 1 : 0;
    if (!unpriced && !chains.empty()) pulsar_spec_keep_picked(s, n, reqs, n);
    return true;
}

/* ---- the round API's drafter ---------------------------------------------------------------------------------- */

/* The committed rows of the last verify, any number of banks (each bank's rows one run, in the step's row order):
 * their context, on the decode arms (they are verify rows). */
bool block_absorb_banked(pulsar_session *s, const uint32_t *rows, const uint32_t *banks, const int32_t *, uint32_t n) {
    const pulsar_state_set *ss = s->states;
    if (!LANE(s)) return fail("an absorb with no block lane");
    int32_t pos[PULSAR_DRIVER_HEADS_MAX], bank[PULSAR_DRIVER_HEADS_MAX];
    if (n == 0) return true;
    if (n > PULSAR_DRIVER_HEADS_MAX) return fail("an absorb wider than the verify");
    /* one call a bank's run of consecutive step rows (the taps are addressed by the step row) */
    for (uint32_t i = 0; i < n;) {
        const uint32_t b = banks[i];
        uint32_t j = i + 1;
        while (j < n && banks[j] == b) j++;
        bool run = b < ss->n_banks && ss->bank_pos[b] >= j - i;
        for (uint32_t r = i + 1; run && r < j; r++) run = rows[r] == rows[r - 1] + 1u;
        if (!run) return fail("a bank's absorbed rows are not one run of its kept rows");
        const uint32_t p0 = ss->bank_pos[b] - (j - i);   /* the commit moved the counter past the kept rows */
        for (uint32_t r = i; r < j; r++) {
            pos[r] = (int32_t)(p0 + (r - i));
            bank[r] = (int32_t)b;
        }
        if (!absorb_rows(s, rows[i], pos + i, bank + i, j - i, false)) return false;
        i = j;
    }
    return true;
}

uint32_t block_draft(pulsar_session *s, int next_base, bool, float temperature, int top_k, float top_p, float min_p,
                     uint64_t *rng) {
    spec_redraft_req rec;
    memset(&rec, 0, sizeof rec);
    pulsar_spec_draft_req_init(s, &rec, next_base, temperature, top_k, top_p, min_p);
    char err[200];
    spec_redraft_req *req = &rec;
    const uint32_t b = s->states->live_bank;
    if (!draft_chains(s, &b, &rng, &req, 1, err, sizeof err)) {
        /* L306: the round refuses, as the batched lane's does (block_draft_batch) -- no plain round */
        fprintf(stderr, "pulsar: block drafter: %s -- the round refuses\n", err);
        pulsar_spec_drop_pendings(&s->spec);
        return PULSAR_DRAFT_REFUSED;
    }
    pulsar_spec_redraft_stamp(s, &rec);
    return rec.keep;
}

int block_draft_batch(pulsar_session *s, pulsar_spec_round **rounds, const uint32_t *banks, uint64_t **rngs, int n,
                      char *err, size_t errlen) {
    std::vector<uint32_t> b;
    std::vector<uint64_t *> r;
    std::vector<spec_redraft_req *> q;
    for (int i = 0; i < n; i++) {
        spec_redraft_req *rq = &rounds[i]->redraft;
        if (!rq->valid || rq->n_draft == 0) continue;
        if (banks[i] >= s->states->n_banks) {
            snprintf(err, errlen, "the draft's bank %u is outside the pool", banks[i]);
            return -1;
        }
        b.push_back(banks[i]);
        r.push_back(rngs[i]);
        q.push_back(rq);
    }
    if (!b.empty() && !draft_chains(s, b.data(), r.data(), q.data(), (uint32_t)b.size(), err, errlen)) return -1;
    for (spec_redraft_req *rq : q) rq->done = true;
    return 0;
}

/* L306: the draft pass's graph catalog, at the startup warmup (pulsar_drafter_ops::warm).  A live pass's graph key is
 * (lane, banks, bound): every bank count a pass carries (1 .. kDraftRows / block) at every attention bound a pass
 * reaches (256, doubling, to the first above the deepest pass's last row, ctx - 2 + block - 1), each twice -- the
 * first use warms the pass's scratch, the second is captured inside the warmup's segment window.  On the installed
 * bank (repeated for the bank counts: a pass reads each bank's ring and anchor, never writes them) at a stand-in
 * position per bound: a pass writes only the lane's arena, so a deeper position than the bank holds reads its ring
 * (position-addressed, modulo its window) without touching any state. */
int block_warm(pulsar_session *s, char *err, size_t errlen) {
    pulsar_block_lane *l = LANE(s);
    if (!l || dump_dir()) return 0;   /* no lane; or the dump's eager passes: nothing is captured */
    const pulsar_block_drafter *d = l->d;
    const pulsar_state_set *ss = s->states;
    const uint32_t B = d->block, per = kDraftRows / B, bank = ss->live_bank;
    if (ss->bank_pos[bank] == 0 || ss->ctx < 2u + B) {
        snprintf(err, errlen, "the installed bank holds no context to warm the draft pass from");
        return -1;
    }
    const uint64_t last_max = (uint64_t)ss->ctx - 2u + B - 1u;
    uint32_t top = 256u;
    while (top <= last_max) top *= 2u;
    pulsar_gpu_seg_stats a, b;
    pulsar_gpu_seg_stats_get(&a);
    const double t0 = now_sec();
    uint32_t banks[kDraftRows], pos[kDraftRows], shapes = 0, n_bounds = 0;
    int32_t carry[kDraftRows];
    for (uint32_t bound = 256u; bound <= top; bound *= 2u, n_bounds++) {
        /* a pass's last row P + block - 1 lands in [bound / 2, bound) (in [0, 256) for the first) */
        const uint32_t P = (uint32_t)std::min<uint64_t>((uint64_t)bound - B, (uint64_t)ss->ctx - 2u);
        for (uint32_t nb = 1; nb <= per; nb++, shapes++) {
            for (uint32_t i = 0; i < nb; i++) {
                banks[i] = bank;
                pos[i] = P;
                carry[i] = 0;
            }
            for (int use = 0; use < 2; use++)
                if (!draft_pass(s, banks, pos, carry, nb)) {
                    snprintf(err, errlen, "the draft pass of %u bank(s) at bound %u refused at the warmup", nb, bound);
                    return -1;
                }
        }
    }
    pulsar_gpu_seg_stats_get(&b);
    /* rule 5: the shapes the warmup declared for this lane and what was captured of them */
    fprintf(stderr, "pulsar: block drafter: L306 warmup: the draft pass at %u bank count(s) x %u attention bounds "
                    "(256 .. %u) = %u shapes, %llu graphs captured (%.1f MiB) in %.2f s\n",
            per, n_bounds, top, shapes, (unsigned long long)(b.captures - a.captures),
            (double)(b.graph_bytes > a.graph_bytes ? b.graph_bytes - a.graph_bytes : 0u) / 1048576.0, now_sec() - t0);
    return 0;
}

}  // namespace

/* The drafter's functions; its numbers are set per declaration in its copy (pulsar_block_drafter_read: the chain is the
 * block's block - 1 slots, the stop rule's tau measured), no adaptive depth (the stop rule and the priced width end a
 * chain where it no longer pays). */
const pulsar_drafter_ops PULSAR_BLOCK_DRAFTER_OPS = {
    /* .name          = */ "block",
    /* .depth         = */ 0u,
    /* .depth_max     = */ 0u,
    /* .tau           = */ 0.0f,
    /* .adapt         = */ NULL,
    /* .prime         = */ NULL,   /* the context is absorbed with the trunk's every step */
    /* .absorb        = */ NULL,   /* a round's rows together: absorb_banked */
    /* .absorb_banked = */ block_absorb_banked,
    /* .draft         = */ block_draft,
    /* .harvest       = */ NULL,
    /* .draft_batch   = */ block_draft_batch,
    /* .warm          = */ block_warm,   /* L306: its draft pass's bank counts x bounds, at the warmup */
};

bool pulsar_drafter_lane_absorb(pulsar_session *s, pulsar_step_mode mode, const int32_t *tokens, const int32_t *pos,
                                const int32_t *bank, uint32_t n, uint32_t row0, bool eval) {
    if (s->driver && s->driver->mtp) return pulsar_mtp_absorb(s, mode, tokens, pos, bank, n, row0, eval);
    if (!LANE(s)) return true;
    /* PREFILL: one bank's prompt run (the prompt arms); DECODE: one row a bank (the decode arms) */
    return absorb_rows(s, row0, pos, bank, n, mode == PULSAR_STEP_PREFILL);
}

/* spec_block.cpp -- the block drafter (spec_block.h; L298): its declaration and binding from the container, its
 * declared state, the context absorb, the block forward and the chain, behind the speculation round API
 * (session_spec.cpp) and the core driver's verify target (step_driver.cpp PULSAR_DRIVER_SPEC_TARGET). */
#include "pulsar_engine_internal.h"
#include "spec_block.h"
#include "spec_mtp.h"
#include "cuda/pulsar_cuda_attn_gqa.h"
#include "cuda/pulsar_cuda_blockdraft.h"
#include "cuda/pulsar_cuda_head.h"
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

bool pulsar_block_drafter_read(const pulsar_model *m, pulsar_block_drafter *d) {
    *d = pulsar_block_drafter{};
    d->kind = "lilicorr";
    bool ok = get_u32(m, "lilicorr.hidden_size", &d->hidden);
    ok = get_u32(m, "lilicorr.num_hidden_layers", &d->n_layer) && ok;
    ok = get_u32(m, "lilicorr.num_attention_heads", &d->n_head) && ok;
    ok = get_u32(m, "lilicorr.num_key_value_heads", &d->n_kv) && ok;
    ok = get_u32(m, "lilicorr.head_dim", &d->head_dim) && ok;
    ok = get_u32(m, "lilicorr.intermediate_size", &d->ff) && ok;
    ok = get_u32(m, "lilicorr.vocab_size", &d->vocab) && ok;
    ok = get_u32(m, "lilicorr.block_size", &d->block) && ok;
    ok = get_f32(m, "lilicorr.rms_norm_eps", &d->eps) && ok;
    ok = get_f32(m, "lilicorr.rope_theta", &d->rope_theta) && ok;
    uint32_t mask = 0;
    ok = get_u32(m, "lilicorr.dflash.mask_token_id", &mask) && ok;
    d->mask_id = (int32_t)mask;
    ok = get_u32(m, "lilicorr.dflash.conv_kernel_size", &d->conv_taps) && ok;
    ok = get_u32(m, "lilicorr.dflash.conv_group_size", &d->conv_group) && ok;
    ok = get_u32(m, "lilicorr.dflash.swa_window_size", &d->window) && ok;
    ok = get_u32(m, "lilicorr.dflash.lilicorr_candidate_topk", &d->topk) && ok;
    ok = want_bool(m, "lilicorr.dflash.causal", false) && ok;   /* bidirectional inside the block */
    ok = want_bool(m, "lilicorr.dflash.use_swa", true) && ok;   /* every layer windowed over the context */
    ok = get_ids(m, "lilicorr.dflash.target_layer_ids", d->tap_block, PULSAR_BLOCK_MAX_TAPS, &d->n_taps) && ok;
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
    if (why) return refuse("the declaration (lilicorr.*) is outside what the blocks implement: %s%s", why);
    return true;
}

bool pulsar_block_drafter_bind(pulsar_role_binder *b, pulsar_block_drafter *d) {
    const pulsar_role_admit A_BF16 = {PULSAR_ROLE_PLAIN(PULSAR_TENSOR_BF16), false, PULSAR_ROLE_DENSE, 0};
    const pulsar_role_admit A_LINEAR = {0, true, PULSAR_ROLE_DENSE, PULSAR_ACTS(PULSAR_ACT_ROWS_BF16)};
    const uint64_t E = d->hidden, Q = (uint64_t)d->n_head * d->head_dim, KV = (uint64_t)d->n_kv * d->head_dim;
    const uint64_t CP = 2ull * d->conv_taps * (E / d->conv_group);
    auto top = [&](const char *role) { return pulsar_role_in(role, BLOCK, -1); };
    d->fc = pulsar_role_bind(b, top("fc"), A_LINEAR, 2, (uint64_t)d->n_taps * E, E);
    d->hidden_norm = pulsar_role_bind(b, top("hidden_norm"), A_BF16, 1, E);
    d->out_norm = pulsar_role_bind(b, top("out_norm"), A_BF16, 1, E);
    d->head = pulsar_role_bind(b, top("head"), A_LINEAR, 2, E, d->vocab);
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
    L.kv_fmt = PULSAR_GQA_KV_BF16;
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
    pulsar_state_decl anchor{"draft_anchor", PULSAR_STATE_FIXED, PULSAR_STATE_CAPTURE_NONE, false, false, 0,
                             (uint64_t)d->hidden * 2u, 0, 0, NULL, NULL, NULL};
    d->decl_kv = pulsar_state_plan_add(p, kv);
    d->decl_anchor = pulsar_state_plan_add(p, anchor);
    for (uint32_t l = 0; l < d->n_layer; l++) p->slot[p->n_slots + l] = 1u << d->decl_kv;
    p->slot[p->n_slots] |= 1u << d->decl_anchor;
    p->n_slots += d->n_layer;
    /* no lane counters to keep with the walk: the ring and the anchor ARE the drafter's state at the bank's frontier
     * (a draft writes neither) */
    p->drafter_at = NULL;
    p->drafter_frontier = NULL;
}

void pulsar_block_drafter_describe(const pulsar_block_drafter *d, char *out, size_t cap) {
    int n = snprintf(out, cap, "%s block drafter: %u layers %u/%u/%u, ff %u, block %u, window %u, conv %ux/%u, taps",
                     d->kind, d->n_layer, d->n_head, d->n_kv, d->head_dim, d->ff, d->block, d->window, d->conv_taps,
                     d->conv_group);
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
    uint64_t ids, h, x, cx, t, ob, coef, q, k, v, a, g, u, m, stage, ws, ws_bytes, gqa, gqa_bytes, logits, top_id,
        top_lp, total;
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
    o += a256(R * std::max<uint64_t>(std::max<uint64_t>(2u * F, Q + 2u * KV), std::max<uint64_t>(d->vocab, CP)) * 4u);
    const int qkv[3] = {(int)Q, (int)KV, (int)KV}, gu[2] = {(int)F, (int)F}, eo[1] = {(int)E}, cp[1] = {(int)CP};
    L.ws = o;
    L.ws_bytes = std::max(std::max(multi_ws((int)E, qkv, 3, R), multi_ws((int)E, gu, 2, R)),
                          std::max(std::max(multi_ws((int)F, eo, 1, R), multi_ws((int)Q, eo, 1, R)),
                                   multi_ws((int)E, cp, 1, R)));
    o += L.ws_bytes;
    const pulsar_gqa_layer g = gqa_decl(d, NULL, NULL, true);
    L.gqa = o; L.gqa_bytes = a256(pulsar_gqa_workspace_bytes(&g, (uint32_t)R, d->window + PULSAR_BLOCK_SIZE_MAX));
    o += L.gqa_bytes;
    L.logits = o; o += a256(R * d->vocab * 2u);   /* bf16: the head's rows as its producer emits them */
    L.top_id = o; o += a256(R * d->topk * 4u);
    L.top_lp = o; o += a256(R * d->topk * 4u);
    L.total = o;
    return L;
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
    std::vector<int32_t> top_id;
    std::vector<float> top_lp;
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
    l->top_id.resize((size_t)kDraftRows * d->topk);
    l->top_lp.resize((size_t)kDraftRows * d->topk);
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
    delete l;
}

bool pulsar_block_tap(pulsar_driver_state *ds, uint32_t il, uint32_t sub, const uint16_t *h, uint64_t ld, uint32_t n) {
    pulsar_block_lane *l = ds ? ds->block : NULL;
    if (!l) return true;
    const pulsar_block_drafter *d = l->d;
    bool ok = true;
    for (uint32_t j = 0; ok && j < d->n_taps; j++) {
        if (d->tap_layer[j] != il || d->tap_sub[j] != sub) continue;
        if (n > l->max_rows) return fail("a trunk step wider than the tap rows");
        ok = pulsar_blockdraft_rows_copy_launch(h, ld, n, d->hidden, (uint16_t *)dptr(l->taps) + (uint64_t)j * d->hidden,
                                                l->tap_width, 0) == 0 ||
             fail("the tap copy refused");
    }
    return ok;
}

namespace {

pulsar_block_lane *LANE(const pulsar_session *s) { return s->driver ? s->driver->block : NULL; }

/* ---- the reference grade's dump (diagnostic: PULSAR_BLOCK_DUMP=<dir>, read once) ------------------------------------
 * What tools/blockdraft/ref.py replays against the PyTorch reference of the drafter: every absorbed context row (its
 * bank, position and taps) to <dir>/ctx.bin, and the first PULSAR_BLOCK_DUMP_PASSES (default 8) draft passes (their
 * banks, carries, positions, each row's final hidden and top-k) to <dir>/draft.bin.  Little-endian records:
 *   ctx    u32 n, i32 pos[n], i32 bank[n], u32 width, bf16 taps[n][width]
 *   draft  u32 nb, u32 block, u32 hidden, u32 topk, i32 bank[nb], i32 carry[nb], i32 P[nb],
 *          bf16 o[nb block][hidden], i32 ids[nb block][topk], f32 lp[nb block][topk] */
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
    uint8_t *base = (uint8_t *)dptr(l->arena);
    const absorb_layout &A = l->al;
    uint16_t *c = (uint16_t *)(base + A.c), *cn = (uint16_t *)(base + A.cn), *kb = (uint16_t *)(base + A.kb),
             *vb = (uint16_t *)(base + A.vb);
    float *stage = (float *)(base + A.stage);
    const uint16_t *hn = (const uint16_t *)wp(s, d->hidden_norm, "drafter hidden_norm");
    pulsar_gpu_tensor *anchor = pulsar_state_tensor(ss, ss->plan->n_trunk, d->decl_anchor);
    if (!hn || !anchor) return fail("the drafter's context norm or anchor state is missing");
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
        /* each bank's last row of this pass is its anchor (a later pass's overwrites it) */
        for (uint32_t i = 0; i < m; i++) {
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
 * One pass over nb <= kDraftRows / block banks: the block forward, the head over every row, the top-k of every row
 * read back.  Bank i's rows are [i block, (i + 1) block) at positions P_i ... */
bool draft_pass(pulsar_session *s, const uint32_t *banks, const int32_t *carry, uint32_t nb) {
    pulsar_block_lane *l = LANE(s);
    const pulsar_block_drafter *d = l->d;
    const pulsar_state_set *ss = s->states;
    const uint32_t B = d->block, R = nb * B, E = d->hidden, Q = d->n_head * d->head_dim, KV = d->n_kv * d->head_dim,
                   F = d->ff, CP = 2u * d->conv_taps * (E / d->conv_group), V = d->vocab, half = d->conv_taps * (E / d->conv_group);
    if (R > kDraftRows || nb == 0) return fail("a draft pass outside the lane's rows");
    uint8_t *base = (uint8_t *)dptr(l->arena);
    const draft_layout &D = l->dl;
    uint16_t *h = (uint16_t *)(base + D.h), *x = (uint16_t *)(base + D.x), *cx = (uint16_t *)(base + D.cx),
             *t = (uint16_t *)(base + D.t), *ob = (uint16_t *)(base + D.ob), *coef = (uint16_t *)(base + D.coef),
             *q = (uint16_t *)(base + D.q), *k = (uint16_t *)(base + D.k), *v = (uint16_t *)(base + D.v),
             *a = (uint16_t *)(base + D.a), *g = (uint16_t *)(base + D.g), *u = (uint16_t *)(base + D.u),
             *mm = (uint16_t *)(base + D.m);
    float *stage = (float *)(base + D.stage);
    uint16_t *logits = (uint16_t *)(base + D.logits);
    void *ws = base + D.ws;
    /* the rows: [carry, mask x (B - 1)] per bank */
    int32_t tok[kDraftRows];
    for (uint32_t i = 0; i < nb; i++) {
        const uint32_t P = ss->bank_pos[banks[i]];
        tok[i * B] = carry[i];
        for (uint32_t j = 1; j < B; j++) tok[i * B + j] = d->mask_id;
        for (uint32_t j = 0; j < B; j++) {
            l->row_seq[i * B + j] = i;
            l->row_pos[i * B + j] = P + j;
        }
        if (carry[i] < 0 || (uint32_t)carry[i] >= V) return fail("a carry outside the vocabulary");
    }
    bool ok = pulsar_gpu_tensor_write_async(l->arena, D.ids, tok, (uint64_t)R * 4u) != 0;
    pulsar_head_decl hd{};
    hd.n_embd = E;
    hd.n_logits = V;
    hd.n_real = V;
    hd.logit_scale = 1.0f;
    const uint16_t *table = (const uint16_t *)wp(s, d->embed, "drafter embed (the target's)");
    ok = ok && table &&
         pulsar_head_embed_launch(&hd, table, V, (const int32_t *)(base + D.ids), R, h, E, 0) == 0;
    if (!ok) return fail("the block's embedding refused");
    for (uint32_t li = 0; li < d->n_layer; li++) {
        const pulsar_block_layer &L = d->layer[li];
        const uint16_t *in_n = (const uint16_t *)wp(s, L.in_norm, "drafter in_norm");
        const uint16_t *post_n = (const uint16_t *)wp(s, L.post_norm, "drafter post_norm");
        const uint16_t *ab = (const uint16_t *)wp(s, L.ac_base, "drafter attn conv base");
        const uint16_t *mb = (const uint16_t *)wp(s, L.mc_base, "drafter mlp conv base");
        if (!in_n || !post_n || !ab || !mb) return false;
        /* x = rmsnorm(h) (layer 0), else h += the previous layer's output and x its norm */
        ok = li == 0 ? pulsar_rows_rmsnorm_launch(h, E, R, E, in_n, d->eps, x, E, 0) == 0
                     : pulsar_rows_add_launch(h, E, t, E, R, E, in_n, d->eps, x, E, 0) == 0;
        if (!ok) return fail("the input norm refused");
        /* the attention conv's taps, its prepare */
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
            if (!linears(s, 3, w, (int)E, out, "drafter q / k / v", cx, R, false, stage, yb, ws, D.ws_bytes)) return false;
        }
        for (uint32_t i = 0; i < nb; i++) l->seqs[i] = ring_of(s, d, li, banks[i]);
        const pulsar_gqa_layer G =
            gqa_decl(d, (const uint16_t *)wp(s, L.q_norm, "drafter q_norm"), (const uint16_t *)wp(s, L.k_norm, "drafter k_norm"), true);
        pulsar_gqa_io io{};
        io.q = {q, Q, d->head_dim};
        io.k = {k, KV, d->head_dim};
        io.v = {v, KV, d->head_dim};
        io.out_bf16 = a;
        io.ld_out = Q;
        io.workspace = base + D.gqa;
        io.workspace_bytes = D.gqa_bytes;
        if (!G.q_norm_w || !G.k_norm_w ||
            pulsar_gqa_forward(&G, l->seqs.data(), nb, l->row_seq.data(), l->row_pos.data(), R, &io, 0) != 0)
            return fail("the block attention refused");
        {
            const pulsar_tensor *w[1] = {L.o};
            const int out[1] = {(int)E};
            uint16_t *yb[1] = {ob};
            if (!linears(s, 1, w, (int)Q, out, "drafter o_proj", a, R, false, stage, yb, ws, D.ws_bytes)) return false;
        }
        /* the conv's finish (side 1: its taps and base), the residual add, the MLP's pre-norm */
        if (pulsar_blockdraft_conv_launch(ob, E, coef + half, CP, ab + (uint64_t)d->conv_taps * E, d->conv_taps,
                                          d->conv_group, B, R, E, t, E, 0) != 0 ||
            pulsar_rows_add_launch(h, E, t, E, R, E, post_n, d->eps, x, E, 0) != 0)
            return fail("the attention's finish refused");
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
            if (!linears(s, 2, w, (int)E, out, "drafter gate / up", cx, R, false, stage, yb, ws, D.ws_bytes)) return false;
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
    /* o = rmsnorm(h + t) * norm, the head over every row, the top-k */
    const uint16_t *on = (const uint16_t *)wp(s, d->out_norm, "drafter norm");
    if (!on || pulsar_rows_add_launch(h, E, t, E, R, E, on, d->eps, x, E, 0) != 0) return fail("the final norm refused");
    {
        const pulsar_tensor *w[1] = {d->head};
        const int out[1] = {(int)V};
        uint16_t *yb[1] = {logits};
        if (!linears(s, 1, w, (int)E, out, "drafter lm_head", x, R, false, stage, yb, ws, D.ws_bytes)) return false;
    }
    if (pulsar_rows_topk_logprob_launch(logits, V, R, V, d->topk, (int32_t *)(base + D.top_id),
                                        (float *)(base + D.top_lp), 0) != 0)
        return fail("the candidates' top-k refused");
    if (!pulsar_gpu_tensor_read(l->arena, D.top_id, l->top_id.data(), (uint64_t)R * d->topk * 4u) ||
        !pulsar_gpu_tensor_read(l->arena, D.top_lp, l->top_lp.data(), (uint64_t)R * d->topk * 4u))
        return fail("could not read the candidates");
    if (dump_dir()) {
        static int passes = -1;
        if (passes < 0) {
            const char *v = getenv("PULSAR_BLOCK_DUMP_PASSES");
            passes = v && v[0] ? atoi(v) : 8;
        }
        FILE *f = passes > 0 ? dump_open("draft.bin") : NULL;
        std::vector<uint16_t> o((size_t)R * E);
        if (f && pulsar_gpu_tensor_read(l->arena, D.x, o.data(), o.size() * 2u)) {
            passes--;
            int32_t bk[kDraftRows], P[kDraftRows];
            for (uint32_t i = 0; i < nb; i++) {
                bk[i] = (int32_t)banks[i];
                P[i] = (int32_t)ss->bank_pos[banks[i]];
            }
            const uint32_t hdr[4] = {nb, B, E, d->topk};
            fwrite(hdr, 4, 4, f);
            fwrite(bk, 4, nb, f);
            fwrite(carry, 4, nb, f);
            fwrite(P, 4, nb, f);
            fwrite(o.data(), 2, o.size(), f);
            fwrite(l->top_id.data(), 4, (size_t)R * d->topk, f);
            fwrite(l->top_lp.data(), 4, (size_t)R * d->topk, f);
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
    uint32_t pb[kDraftRows], pi[kDraftRows];
    int32_t pc[kDraftRows];
    std::vector<float> row;
    std::vector<uint16_t> row16;
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
            pc[nb] = q->next_base;
            pi[nb++] = i;
        }
        i0 = i;
        if (nb == 0) continue;
        if (!draft_pass(s, pb, pc, nb)) {
            snprintf(err, errlen, "the block draft failed (see the log)");
            return false;
        }
        for (uint32_t b = 0; b < nb; b++) {
            spec_redraft_req *q = reqs[pi[b]];
            const uint32_t P = ss->bank_pos[pb[b]];
            uint32_t K = std::min(q->n_draft, B - 1u);
            if ((uint64_t)P + 1u + K > ss->ctx) K = (uint32_t)(ss->ctx - P - 1u);
            for (uint32_t j = 1; j <= K; j++) {
                const uint32_t r = b * B + j;
                const int32_t top = l->top_id[(size_t)r * d->topk];
                q->conf[j - 1] = expf(l->top_lp[(size_t)r * d->topk]);
                if (q->temperature <= 0.0f) {
                    q->refined[j] = top;
                    continue;
                }
                /* sampled: the row under the request's params, drawn at the draft's key (j places past the carry) */
                row16.resize(V);
                row.resize(V);
                if (!pulsar_gpu_tensor_read(l->arena, l->dl.logits + (uint64_t)r * V * 2u, row16.data(), (uint64_t)V * 2u)) {
                    snprintf(err, errlen, "could not read a draft row");
                    return false;
                }
                for (uint32_t c = 0; c < V; c++) {   /* bf16 -> f32 is exact */
                    const uint32_t bits = (uint32_t)row16[c] << 16;
                    memcpy(&row[c], &bits, 4);
                }
                pulsar_sample_dist qd{};
                if (!pulsar_sample_dist_build(row.data(), V, q->temperature, q->top_k, q->top_p, q->min_p,
                                              &s->sample_scratch, &qd)) {
                    snprintf(err, errlen, "a draft row has no distribution");
                    return false;
                }
                const int dt = pulsar_sample_dist_draw(&qd, pulsar_sample_key(*rngs[pi[b]], j));
                pulsar_sample_dist_free(&qd);
                if (dt < 0) {
                    snprintf(err, errlen, "a draft draw failed");
                    return false;
                }
                q->refined[j] = dt;
            }
            q->keep = K;
        }
    }
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
        fprintf(stderr, "pulsar: block drafter: %s (the round takes a plain step next)\n", err);
        pulsar_spec_drop_pendings(&s->spec);
        return 0;
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

}  // namespace

/* The drafter's numbers: a chain is the block's slots (block - 1: LiLiCorr's 7), every slot kept (no stop rule: the
 * block drafts every slot in its one pass, so a slot costs only its verify row), no adaptive depth. */
const pulsar_drafter_ops PULSAR_BLOCK_DRAFTER_OPS = {
    /* .name          = */ "block",
    /* .depth         = */ PULSAR_BLOCK_SIZE_MAX - 1u,
    /* .depth_max     = */ PULSAR_BLOCK_SIZE_MAX - 1u,
    /* .tau           = */ 0.0f,
    /* .adapt         = */ NULL,
    /* .prime         = */ NULL,   /* the context is absorbed with the trunk's every step */
    /* .absorb        = */ NULL,   /* a round's rows together: absorb_banked */
    /* .absorb_banked = */ block_absorb_banked,
    /* .draft         = */ block_draft,
    /* .harvest       = */ NULL,
    /* .draft_batch   = */ block_draft_batch,
};

bool pulsar_drafter_lane_absorb(pulsar_session *s, pulsar_step_mode mode, const int32_t *tokens, const int32_t *pos,
                                const int32_t *bank, uint32_t n, uint32_t row0, bool eval) {
    if (s->driver && s->driver->mtp) return pulsar_mtp_absorb(s, mode, tokens, pos, bank, n, row0, eval);
    if (!LANE(s)) return true;
    /* PREFILL: one bank's prompt run (the prompt arms); DECODE: one row a bank (the decode arms) */
    return absorb_rows(s, row0, pos, bank, n, mode == PULSAR_STEP_PREFILL);
}

/* nemotron_parse_forward.cpp -- Parse's forward and its generation (nemotron_parse.h section 4).
 *
 * Every kernel is a shared block's; what is Parse's here is the composition:
 *
 *   tower    res = [class tokens | pos + patch(x)]  -- the position rows (host, cached per page geometry) seed the f32
 *                                                     residual, the patch Linear's f32 output added
 *            27 x { xn = LN1(res);  qkv = xn Wqkv + b                     the f16 rows arm; the rows bias emit
 *                   a = MHA(q, k, v) -- 16 heads of 72, every row of the page, 1/sqrt(72)   pulsar_cuda_fmha.h
 *                   res += a Wproj + b, xn = LN2(res)                     pulsar_rows_res32_add (its LayerNorm fused)
 *                   m = GELU(xn Wfc1 + b);  res += m Wfc2 + b, xn = LN1'(res) }
 *   neck     p = LN1(patches Wconv1 + b);  merged = LN2([p_4c | .. | p_4c+3] Wconv2)   (four rows ARE one row of the
 *            merge's input); summary = LN3([cls_0 | cls_1] Wsum + b)  -> the prefix [merged | summary]
 *   decoder  h = LN_emb([prefix | 32 * E[prompt]]);  10 x { q, k, v = LN(h) W + b;  a = GQA(q, k, v) -- 16q / 8kv of
 *            64, causal over the whole sequence, 1/8, no rope, the KV pool per layer;  h += a Wo + b;
 *            h += GELU(LN(h) Wfc1 + b) Wfc2 + b };  logits = LN(h) E^T   (the tied head)
 *   decode   one row per step at the next position: the same layers on the pool, the head on that row
 *   sample   greedy over the logits after the repetition penalty (each id already in the sequence -- the prompt and
 *            the generated: l < 0 ? l * 1.1 : l / 1.1), the first maximum; until eos, the new-token cap, or the
 *            model's context
 *
 * THE FORMATS.  The residuals (the tower's and the decoder's) are f32: RADIO's late layers grow a few very-high-norm
 * tokens, and a bf16 residual drops the small updates the others get every layer (measured: vit_out row cosine down
 * to 0.08).  Every Linear's operand is f16 rows written by its producer (rule 3: the LayerNorms, the GELU, the fused
 * attention, the page's patches, the tower's output for the neck) -- 11 significant bits, as HF's own fp16 run --
 * except the decoder attention's: the shared GQA block reads and writes bf16 rows, so q / k / v are emitted bf16 and
 * Wo reads bf16 (the f16 arm's exact bf16 prescale).  The prefix and the embeddings are f32 rows (the f16 table times
 * 32 is exact).  Where pulsar rounds differently from the f32 reference it is graded (retriever/parse/
 * grade_stages.py): the f16 weights, the f16 operand rows, the decoder's bf16 attention rows, FMHA's and F1's
 * softmaxes.
 *
 * Linears run in row chunks of NP_CHUNK on the prompt arms (a row's bytes do not depend on its chunk), a decode row
 * on the decode arm.
 *
 * PULSAR_PARSE_DUMP=DIR writes the stages the reference records (vit_in, vit_mid, vit_out, prefix, emb, hidden,
 * logits0 -- f32 little-endian, row-major) for the stage grade; nothing else reads them. */
#include "nemotron_parse.h"
#include "engine/exl3_trellis.h"   /* exl3_f16_to_f32: the f16 stage dump */
#include "cuda/pulsar_cuda_attn_gqa.h"
#include "cuda/pulsar_cuda_f16_rows.h"
#include "cuda/pulsar_cuda_fmha.h"
#include "cuda/pulsar_cuda_qwen.h"     /* the rows front door's launchers (family-neutral) */
#include "cuda/pulsar_cuda_rowops.h"

#include <math.h>
#include <pthread.h>
#include <time.h>

#include <optional>

namespace {

const char *const ARCH = PULSAR_NEMOTRON_PARSE_ARCH;
constexpr uint32_t NP_CHUNK = 2048;
constexpr uint32_t NP_PROMPT_MAX = 64;
constexpr pulsar_rows_out F16 = PULSAR_ROWS_OUT_F16;

double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return 1e3 * (double)t.tv_sec + 1e-6 * (double)t.tv_nsec;
}

enum np_buf {
    B_PATCH, B_RES, B_XN, B_QKV, B_ATT, B_MLP, B_STAGE, B_LWS,                    /* the tower (B_RES f32) */
    B_EMB, B_H, B_DXN, B_Q, B_K, B_V, B_DATT, B_DAWS, B_IDS, B_LOGITS, B_POOLS,   /* the decoder (B_EMB, B_H f32) */
    B_COUNT
};

}  // namespace

struct pulsar_nemotron_parse_run {
    pulsar_gqa_layer dec;
    uint32_t n_max;            ///< the tower's rows at the largest page: class tokens + patches
    uint32_t s_max;            ///< the decoder's prefill rows at the largest page: prefix + a prompt of NP_PROMPT_MAX
    pulsar_gpu_tensor *t[B_COUNT];
    uint64_t bytes[B_COUNT];
    uint64_t pool_bytes;       ///< one layer's KV pool: max_seq tokens
    const float *cls_rows;     ///< f32 [n_cls][vit_width] (the mapping's)
    uint32_t pos_h, pos_w, pos_ch, pos_cw;   ///< the cached position rows' geometry
    std::vector<float> pos_rows;             ///< f32 [patches][vit_width]
    pthread_mutex_t mu;                      ///< one page at a time on this engine (its KV spans the steps)
};

namespace {

pulsar_nemotron_parse_model *W(const pulsar_engine *e) { return (pulsar_nemotron_parse_model *)e->family_state; }
template <class T = void>
T *dp(const pulsar_gpu_tensor *t) { return (T *)pulsar_gpu_tensor_device_ptr(t); }

bool fail(char *err, size_t errlen, const char *what) {
    if (err) snprintf(err, errlen, "%s: %s", ARCH, what);
    return false;
}

void write_f32(const char *dir, const char *name, const std::vector<float> &v) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s.f32", dir, name);
    if (FILE *f = fopen(path, "wb")) {
        fwrite(v.data(), 4, v.size(), f);
        fclose(f);
    }
}

struct ctx {
    pulsar_engine *e;
    pulsar_nemotron_parse_model *w;
    pulsar_nemotron_parse_run *r;
    const char *dump;
    char *err;
    size_t errlen;
    const void *wp(const pulsar_tensor *t, const char *what) { return pulsar_weight_device_ptr(&e->model, t, what); }
    const float *fp(const pulsar_tensor *t, const char *what) { return (const float *)wp(t, what); }
    float *stage() { return dp<float>(r->t[B_STAGE]); }
    /* the stage [rows][out] f32 = x [rows][in] times t: f16 operand rows (the f16 arm as they are) */
    bool lin(const pulsar_tensor *t, int in, int out, const uint16_t *x_f16, uint32_t rows, bool prompt, const char *what) {
        pulsar_rows_linear l;
        if (!pulsar_linear_rows_ref(&e->model, t, in, out, prompt, what, &l)) return fail(err, errlen, what);
        return pulsar_rows_linear_f16_launch(&l, x_f16, (int)rows, stage(), 0) == 0 || fail(err, errlen, what);
    }
    /* the same over bf16 operand rows (the f16 arm's exact prescale) -- the decoder attention's output */
    bool lin_bf16(const pulsar_tensor *t, int in, int out, const uint16_t *x_bf16, uint32_t rows, bool prompt,
                  const char *what) {
        pulsar_rows_linear l;
        if (!pulsar_linear_rows_ref(&e->model, t, in, out, prompt, what, &l)) return fail(err, errlen, what);
        return pulsar_rows_linear_launch(&l, x_bf16, (int)rows, stage(), dp(r->t[B_LWS]), r->bytes[B_LWS], 0) == 0 ||
               fail(err, errlen, what);
    }
    /* a stage for the grade: f32 rows of a buffer */
    void put(const char *name, const pulsar_gpu_tensor *t, uint64_t rows, uint32_t width) {
        if (!dump) return;
        std::vector<float> out(rows * width);
        (void)pulsar_gpu_tensor_read(t, 0, out.data(), out.size() * 4u);
        write_f32(dump, name, out);
    }
    /* the same of f16 rows, widened */
    void put16(const char *name, const pulsar_gpu_tensor *t, uint64_t rows, uint32_t width) {
        if (!dump) return;
        std::vector<uint16_t> h(rows * width);
        (void)pulsar_gpu_tensor_read(t, 0, h.data(), h.size() * 2u);
        std::vector<float> out(h.size());
        for (size_t i = 0; i < h.size(); i++) out[i] = exl3_f16_to_f32(h[i]);
        write_f32(dump, name, out);
    }
};

/* the tower over a prepared page: res holds the ViT's output (f32), xn its f16 rows, n = class tokens + patches */
bool tower(ctx &c, const pulsar_nemotron_parse_page &pg, uint32_t n_patch) {
    pulsar_nemotron_parse_run *r = c.r;
    const pulsar_nemotron_parse_model *w = c.w;
    const pulsar_nemotron_parse_shape &s = w->shape;
    const uint32_t D = s.vit_width, NC = s.n_cls, N = NC + n_patch, HD = s.vit_head_dim;
    const uint32_t M = s.vit_mlp, P = 3u * s.patch * s.patch;
    float *res = dp<float>(r->t[B_RES]);
    uint16_t *xn = dp<uint16_t>(r->t[B_XN]), *qkv = dp<uint16_t>(r->t[B_QKV]), *att = dp<uint16_t>(r->t[B_ATT]),
             *mlp = dp<uint16_t>(r->t[B_MLP]), *patch = dp<uint16_t>(r->t[B_PATCH]);
    /* the residual's seed: the class tokens, then the position rows (cached for the geometry) */
    if (r->pos_h != pg.h || r->pos_w != pg.w || r->pos_ch != pg.ch || r->pos_cw != pg.cw) {
        pulsar_nemotron_parse_positions(&s, w->pos_field, pg.h, pg.w, pg.ch, pg.cw, &r->pos_rows);
        r->pos_h = pg.h; r->pos_w = pg.w; r->pos_ch = pg.ch; r->pos_cw = pg.cw;
    }
    /* phases (pulsar_group_gpu_phase): the seed through the first LayerNorm, each block, the output's rows */
    std::optional<pulsar_group_gpu_phase> ph;
    ph.emplace();
    if (!pulsar_gpu_tensor_write(r->t[B_RES], 0, r->cls_rows, (uint64_t)NC * D * 4u) ||
        !pulsar_gpu_tensor_write(r->t[B_RES], (uint64_t)NC * D * 4u, r->pos_rows.data(), (uint64_t)n_patch * D * 4u) ||
        !pulsar_gpu_tensor_write(r->t[B_PATCH], 0, pg.patches.data(), (uint64_t)n_patch * P * 2u))
        return fail(c.err, c.errlen, "the page's rows did not reach the device");
    for (uint32_t r0 = 0; r0 < n_patch; r0 += NP_CHUNK) {
        const uint32_t m = n_patch - r0 < NP_CHUNK ? n_patch - r0 : NP_CHUNK;
        if (!c.lin(w->patch, (int)P, (int)D, patch + (size_t)r0 * P, m, true, "parse patch") ||
            pulsar_rows_res32_add_launch(res + (size_t)(NC + r0) * D, D, c.stage(), D, NULL, m, D, NULL, NULL, 0.f, NULL,
                                         0, F16, 0) != 0)
            return fail(c.err, c.errlen, "the patch embedding refused");
    }
    c.put("vit_in", r->t[B_RES], N, D);
    /* the blocks: the shared ViT encoder (engine/vit_encoder.h) on f16 operand rows, each block its own GPU phase */
    struct hooks_ud {
        ctx *c;
        std::optional<pulsar_group_gpu_phase> *ph;
        uint32_t N, D;
    } hu = {&c, &ph, N, D};
    const pulsar_vit_hooks hooks = {
        [](void *ud, uint32_t) {
            hooks_ud *h = (hooks_ud *)ud;
            h->ph->reset();
            h->ph->emplace();
        },
        [](void *ud, uint32_t il) {
            hooks_ud *h = (hooks_ud *)ud;
            if (il == 13) h->c->put("vit_mid", h->c->r->t[B_RES], h->N, h->D);
        },
        &hu};
    const pulsar_vit_dims dims = {D, s.vit_heads, HD, M, s.vit_depth, s.vit_eps};
    const pulsar_vit_bufs bufs = {res, xn, qkv, att, mlp, c.stage(), NULL, 0, NP_CHUNK};
    if (!pulsar_vit_encode(&c.e->model, dims, w->vit, F16, bufs, N, &hooks, c.err, c.errlen)) return false;
    ph.reset();
    ph.emplace();
    c.put("vit_out", r->t[B_RES], N, D);
    /* the neck reads the tower's output as f16 rows (its Linears' operand), into xn */
    if (pulsar_rows_bias_emit_launch(res, D, N, D, NULL, PULSAR_ROWS_ACT_NONE, xn, D, F16, 0) != 0)
        return fail(c.err, c.errlen, "the tower's output did not round to its rows");
    return true;
}

/* the neck: the prefix rows into emb [0, n_prefix) (f32) */
bool neck(ctx &c, uint32_t n_patch, uint32_t *n_prefix) {
    pulsar_nemotron_parse_run *r = c.r;
    const pulsar_nemotron_parse_model *w = c.w;
    const pulsar_nemotron_parse_shape &s = w->shape;
    const uint32_t D = s.vit_width, NC = s.n_cls, E = s.neck_width, G = s.merge;
    /* the tower's output as f16 rows (xn), the projected patches (att's buffer) */
    const uint16_t *x = dp<uint16_t>(r->t[B_XN]);
    uint16_t *p1 = dp<uint16_t>(r->t[B_ATT]);
    float *emb = dp<float>(r->t[B_EMB]);
    for (uint32_t r0 = 0; r0 < n_patch; r0 += NP_CHUNK) {
        const uint32_t m = n_patch - r0 < NP_CHUNK ? n_patch - r0 : NP_CHUNK;
        if (!c.lin(w->neck_proj, (int)D, (int)E, x + (size_t)(NC + r0) * D, m, true, "neck conv1") ||
            pulsar_rows_layernorm32_launch(c.stage(), true, E, c.fp(w->neck_proj_b, "neck conv1 bias"), m, E,
                                           c.fp(w->neck_norm1, "neck norm1"), c.fp(w->neck_norm1_b, "neck norm1"),
                                           s.neck_eps, p1 + (size_t)r0 * E, E, F16, NULL, 0, 0) != 0)
            return fail(c.err, c.errlen, "the neck's projection refused");
    }
    const uint32_t nm = n_patch / G;
    for (uint32_t r0 = 0; r0 < nm; r0 += NP_CHUNK) {
        const uint32_t m = nm - r0 < NP_CHUNK ? nm - r0 : NP_CHUNK;
        if (!c.lin(w->neck_merge, (int)(G * E), (int)E, p1 + (size_t)r0 * G * E, m, true, "neck conv2") ||
            pulsar_rows_layernorm32_launch(c.stage(), true, E, NULL, m, E, c.fp(w->neck_norm2, "neck norm2"),
                                           c.fp(w->neck_norm2_b, "neck norm2"), s.neck_eps, NULL, 0, F16,
                                           emb + (size_t)r0 * E, E, 0) != 0)
            return fail(c.err, c.errlen, "the neck's merge refused");
    }
    /* the summary: class tokens 0 and 1, adjacent rows, read as one row of 2 x the width */
    if (!c.lin(w->summary, (int)(2u * D), (int)E, x, 1, true, "neck summary") ||
        pulsar_rows_layernorm32_launch(c.stage(), true, E, c.fp(w->summary_b, "neck summary bias"), 1, E,
                                       c.fp(w->neck_norm3, "neck norm3"), c.fp(w->neck_norm3_b, "neck norm3"), s.neck_eps,
                                       NULL, 0, F16, emb + (size_t)nm * E, E, 0) != 0)
        return fail(c.err, c.errlen, "the neck's summary refused");
    *n_prefix = nm + 1u;
    c.put("prefix", r->t[B_EMB], *n_prefix, E);
    return true;
}

/* the decoder over rows [pos0, pos0 + n) whose embeddings are emb [0, n): the KV pools extended, logits of the last
 * row into r->t[B_LOGITS] */
bool decoder(ctx &c, uint32_t pos0, uint32_t n, bool prompt) {
    pulsar_nemotron_parse_run *r = c.r;
    const pulsar_nemotron_parse_model *w = c.w;
    const pulsar_nemotron_parse_shape &s = w->shape;
    const uint32_t E = s.width, HD = s.width / s.heads, KV = s.kv_heads * HD, F = s.ffn;
    float *h = dp<float>(r->t[B_H]), *emb = dp<float>(r->t[B_EMB]);
    uint16_t *xn = dp<uint16_t>(r->t[B_DXN]), *q = dp<uint16_t>(r->t[B_Q]), *k = dp<uint16_t>(r->t[B_K]),
             *v = dp<uint16_t>(r->t[B_V]), *att = dp<uint16_t>(r->t[B_DATT]), *mlp = dp<uint16_t>(r->t[B_MLP]);
    constexpr pulsar_rows_out BF16 = PULSAR_ROWS_OUT_BF16;
    if (pulsar_rows_layernorm32_launch(emb, true, E, NULL, n, E, c.fp(w->embed_norm, "embed norm"),
                                       c.fp(w->embed_norm_b, "embed norm"), s.eps, NULL, 0, F16, h, E, 0) != 0)
        return fail(c.err, c.errlen, "the embedding LayerNorm refused");
    if (pos0 == 0) c.put("emb", r->t[B_H], n, E);
    if (pulsar_rows_layernorm32_launch(h, true, E, NULL, n, E, c.fp(w->layer[0].attn_norm, "attn norm"),
                                       c.fp(w->layer[0].attn_norm_b, "attn norm"), s.eps, xn, E, F16, NULL, 0, 0) != 0)
        return fail(c.err, c.errlen, "the first attention LayerNorm refused");
    std::vector<uint32_t> row_seq(n, 0u), row_pos(n);
    for (uint32_t i = 0; i < n; i++) row_pos[i] = pos0 + i;
    pulsar_gqa_io io = {};
    io.q = {q, E, HD};
    io.k = {k, KV, HD};
    io.v = {v, KV, HD};
    io.out_bf16 = att;
    io.ld_out = E;
    io.workspace = dp(r->t[B_DAWS]);
    io.workspace_bytes = r->bytes[B_DAWS];
    for (uint32_t il = 0; il < s.n_layer; il++) {
        const pulsar_nemotron_parse_dec_layer &L = w->layer[il];
        for (uint32_t r0 = 0; r0 < n; r0 += NP_CHUNK) {
            const uint32_t m = n - r0 < NP_CHUNK ? n - r0 : NP_CHUNK;
            const uint16_t *xc = xn + (size_t)r0 * E;
            if (!c.lin(L.q, (int)E, (int)E, xc, m, prompt, "dec q") ||
                pulsar_rows_bias_emit_launch(c.stage(), E, m, E, c.fp(L.q_b, "dec q bias"), PULSAR_ROWS_ACT_NONE,
                                             q + (size_t)r0 * E, E, BF16, 0) != 0 ||
                !c.lin(L.k, (int)E, (int)KV, xc, m, prompt, "dec k") ||
                pulsar_rows_bias_emit_launch(c.stage(), KV, m, KV, c.fp(L.k_b, "dec k bias"), PULSAR_ROWS_ACT_NONE,
                                             k + (size_t)r0 * KV, KV, BF16, 0) != 0 ||
                !c.lin(L.v, (int)E, (int)KV, xc, m, prompt, "dec v") ||
                pulsar_rows_bias_emit_launch(c.stage(), KV, m, KV, c.fp(L.v_b, "dec v bias"), PULSAR_ROWS_ACT_NONE,
                                             v + (size_t)r0 * KV, KV, BF16, 0) != 0)
                return fail(c.err, c.errlen, "the decoder's q / k / v refused");
        }
        pulsar_gqa_seq seq = {};
        seq.kv = (uint8_t *)dp(r->t[B_POOLS]) + (uint64_t)il * r->pool_bytes;
        seq.bytes = r->pool_bytes;
        seq.cap = s.max_seq;
        if (pulsar_gqa_forward(&r->dec, &seq, 1, row_seq.data(), row_pos.data(), n, &io, 0) != 0)
            return fail(c.err, c.errlen, "the decoder's attention refused");
        const bool last = il + 1 == s.n_layer;
        const float *nn = last ? c.fp(w->output_norm, "output norm") : c.fp(w->layer[il + 1].attn_norm, "attn norm");
        const float *nb = last ? c.fp(w->output_norm_b, "output norm") : c.fp(w->layer[il + 1].attn_norm_b, "attn norm");
        for (uint32_t r0 = 0; r0 < n; r0 += NP_CHUNK) {
            const uint32_t m = n - r0 < NP_CHUNK ? n - r0 : NP_CHUNK;
            float *hc = h + (size_t)r0 * E;
            uint16_t *xc = xn + (size_t)r0 * E;
            if (!c.lin_bf16(L.o, (int)E, (int)E, att + (size_t)r0 * E, m, prompt, "dec o") ||
                pulsar_rows_res32_add_launch(hc, E, c.stage(), E, c.fp(L.o_b, "dec o bias"), m, E,
                                             c.fp(L.ffn_norm, "ffn norm"), c.fp(L.ffn_norm_b, "ffn norm"), s.eps, xc, E,
                                             F16, 0) != 0 ||
                !c.lin(L.up, (int)E, (int)F, xc, m, prompt, "dec fc1") ||
                pulsar_rows_bias_emit_launch(c.stage(), F, m, F, c.fp(L.up_b, "dec fc1 bias"), PULSAR_ROWS_ACT_GELU, mlp,
                                             F, F16, 0) != 0 ||
                !c.lin(L.down, (int)F, (int)E, mlp, m, prompt, "dec fc2") ||
                pulsar_rows_res32_add_launch(hc, E, c.stage(), E, c.fp(L.down_b, "dec fc2 bias"), m, E, nn, nb, s.eps, xc,
                                             E, F16, 0) != 0)
                return fail(c.err, c.errlen, "a decoder layer refused");
        }
    }
    if (pos0 == 0) c.put16("hidden", r->t[B_DXN], n, E);
    /* the tied head on the last row (one row: the decode arm), into the logits */
    pulsar_rows_linear l;
    if (!pulsar_linear_rows_ref(&c.e->model, w->embed, (int)E, (int)s.vocab, false, "parse head", &l) ||
        pulsar_rows_linear_f16_launch(&l, xn + (size_t)(n - 1u) * E, 1, dp<float>(r->t[B_LOGITS]), 0) != 0)
        return fail(c.err, c.errlen, "the head refused");
    return true;
}

/* the embedding rows of `ids` (x the scale, f32) into emb at row `at` */
bool gather(ctx &c, const int *ids, uint32_t n, uint32_t at) {
    pulsar_nemotron_parse_run *r = c.r;
    const pulsar_nemotron_parse_shape &s = c.w->shape;
    std::vector<int32_t> v(ids, ids + n);
    if (!pulsar_gpu_tensor_write(r->t[B_IDS], 0, v.data(), (uint64_t)n * 4u)) return fail(c.err, c.errlen, "ids upload");
    return pulsar_rows_gather_f16_launch((const uint16_t *)c.wp(c.w->embed, "parse embed"), s.vocab, s.width,
                                         dp<int32_t>(r->t[B_IDS]), n, s.embed_scale,
                                         dp<float>(r->t[B_EMB]) + (size_t)at * s.width, s.width, 0) == 0 ||
           fail(c.err, c.errlen, "the embedding gather refused");
}

/* greedy after the repetition penalty: transformers' RepetitionPenaltyLogitsProcessor then argmax (the first max) */
int pick(std::vector<float> &logits, const std::vector<int> &seq, float penalty, std::vector<uint8_t> *seen) {
    for (int id : seq) (*seen)[(size_t)id] = 1;
    for (size_t i = 0; i < logits.size(); i++)
        if ((*seen)[i]) logits[i] = logits[i] < 0.0f ? logits[i] * penalty : logits[i] / penalty;
    size_t best = 0;
    for (size_t i = 1; i < logits.size(); i++)
        if (logits[i] > logits[best]) best = i;
    return (int)best;
}

int parse(pulsar_engine *e, const uint8_t *image, size_t len, const pulsar_parse_options *opt, pulsar_tokens *ids_out,
          char **text, pulsar_parse_stats *stats, char *err, size_t errlen) {
    pulsar_nemotron_parse_model *w = W(e);
    pulsar_nemotron_parse_run *r = w->run;
    const pulsar_nemotron_parse_shape &s = w->shape;
    ctx c = {e, w, r, getenv("PULSAR_PARSE_DUMP"), err, errlen};
    pulsar_parse_stats st = {};
    std::vector<int> prompt = w->prompt_ids;
    if (opt && opt->prompt && !pulsar_nemotron_parse_encode_prompt(w->vocab, opt->prompt, &prompt, err, errlen))
        return -2;
    if (prompt.size() > NP_PROMPT_MAX) {
        if (err) snprintf(err, errlen, "%s: a prompt of %zu tokens (at most %u)", ARCH, prompt.size(), NP_PROMPT_MAX);
        return -2;
    }
    const double t0 = now_ms();
    pulsar_nemotron_parse_page pg;
    if (!pulsar_nemotron_parse_page_prepare(&s, image, len, &pg, err, errlen)) return -2;
    const uint32_t n_patch = (pg.h / s.patch) * (pg.w / s.patch);
    st.image_w = pg.cw;
    st.image_h = pg.ch;
    pthread_mutex_lock(&r->mu);
    uint32_t n_prefix = 0;
    bool ok = n_patch + s.n_cls <= r->n_max || fail(err, errlen, "the page is larger than the canvas");
    /* the GPU work in the group's phases (pulsar_group_gpu_phase): the tower's (its own), the neck, the prefill, each
     * decode step; the engine's own lock (r->mu) holds the page's buffers and KV across them */
    ok = ok && tower(c, pg, n_patch);
    {
        pulsar_group_gpu_phase ph;
        ok = ok && neck(c, n_patch, &n_prefix);
    }
    const double t1 = now_ms();
    std::vector<int> seq = prompt;
    std::vector<float> logits(s.vocab);
    std::vector<uint8_t> seen(s.vocab, 0);
    {
        pulsar_group_gpu_phase ph;
        ok = ok && gather(c, prompt.data(), (uint32_t)prompt.size(), n_prefix) &&
             decoder(c, 0, n_prefix + (uint32_t)prompt.size(), true) &&
             (pulsar_gpu_tensor_read(r->t[B_LOGITS], 0, logits.data(), (uint64_t)s.vocab * 4u) ||
              fail(err, errlen, "the logits' read failed"));
        if (ok) c.put("logits0", r->t[B_LOGITS], 1, s.vocab);
    }
    const double t2 = now_ms();
    const uint32_t max_new = opt && opt->max_new_tokens ? opt->max_new_tokens : s.max_new;
    uint32_t n_new = 0;
    while (ok) {
        const int tok = pick(logits, seq, s.rep_penalty, &seen);
        seq.push_back(tok);
        n_new++;
        const uint32_t pos = n_prefix + (uint32_t)seq.size() - 1u;
        if ((uint32_t)tok == s.eos || n_new >= max_new || pos + 1u >= s.max_seq) break;
        pulsar_group_gpu_phase ph;
        ok = gather(c, &tok, 1, 0) && decoder(c, pos, 1, false) &&
             (pulsar_gpu_tensor_read(r->t[B_LOGITS], 0, logits.data(), (uint64_t)s.vocab * 4u) ||
              fail(err, errlen, "the logits' read failed"));
    }
    pthread_mutex_unlock(&r->mu);
    const double t3 = now_ms();
    if (!ok) return -1;
    for (int id : seq) pulsar_tokens_push(ids_out, id);
    const std::string t = pulsar_nemotron_parse_decode(w->vocab, seq.data(), seq.size());
    *text = (char *)xmalloc(t.size() + 1);
    memcpy(*text, t.data(), t.size());
    (*text)[t.size()] = '\0';
    st.prefix_rows = n_prefix;
    st.prompt_tokens = (uint32_t)prompt.size();
    st.generated_tokens = n_new;
    st.finished = (uint32_t)seq.back() == s.eos;
    st.vision_ms = t1 - t0;
    st.prefill_ms = t2 - t1;
    st.decode_ms = t3 - t2;
    if (stats) *stats = st;
    return 0;
}

}  // namespace

const pulsar_family_parser PULSAR_NEMOTRON_PARSE_PARSER = {
    /* .parse = */ parse,
};

void pulsar_nemotron_parse_run_free(pulsar_nemotron_parse_model *w) {
    pulsar_nemotron_parse_run *r = w ? w->run : NULL;
    if (!r) return;
    for (pulsar_gpu_tensor *t : r->t) pulsar_gpu_tensor_free(t);
    pthread_mutex_destroy(&r->mu);
    delete r;
    w->run = NULL;
}

bool pulsar_nemotron_parse_run_create(pulsar_engine *e) {
    pulsar_nemotron_parse_model *w = W(e);
    const pulsar_nemotron_parse_shape &s = w->shape;
    pulsar_nemotron_parse_run *r = new pulsar_nemotron_parse_run();
    w->run = r;   /* unload frees it, whatever this verdict */
    pthread_mutex_init(&r->mu, NULL);
    const uint32_t P = s.patch, D = s.vit_width, HS = s.vit_heads * s.vit_head_dim, M = s.vit_mlp;
    const uint32_t E = s.width, KV = s.kv_heads * (s.width / s.heads), F = s.ffn, C = NP_CHUNK;
    const uint32_t n_patch_max = (s.canvas_h / P) * (s.canvas_w / P);
    r->n_max = s.n_cls + n_patch_max;
    r->s_max = n_patch_max / s.merge + 1u + NP_PROMPT_MAX;
    /* the decoder's attention: 16q / 8kv of 64, causal over the whole sequence, the pool per layer, no rope */
    pulsar_gqa_layer &B = r->dec;
    B.n_q = s.heads;
    B.n_kv = s.kv_heads;
    B.head_dim = s.width / s.heads;
    B.kv_fmt = PULSAR_GQA_KV_BF16;
    B.span = PULSAR_GQA_FULL;
    B.scale = 1.0f / sqrtf((float)(s.width / s.heads));
    r->pool_bytes = (uint64_t)s.max_seq * pulsar_gqa_kv_token_bytes(&B);
    const uint64_t n = r->n_max, sm = r->s_max;
    r->bytes[B_PATCH] = (uint64_t)n_patch_max * 3u * P * P * 2u;
    r->bytes[B_RES] = n * D * 4u;
    r->bytes[B_XN] = n * D * 2u;
    r->bytes[B_QKV] = n * 3u * HS * 2u;
    r->bytes[B_ATT] = n * HS * 2u;
    r->bytes[B_MLP] = (uint64_t)C * (M > F ? M : F) * 2u;
    r->bytes[B_STAGE] = (uint64_t)C * (3u * HS > M ? 3u * HS : M) * 4u;
    r->bytes[B_LWS] = pulsar_f16_rows_workspace_bytes((int)C, (int)E);   /* Wo over bf16 rows: the arm's prescale */
    r->bytes[B_EMB] = r->bytes[B_H] = sm * E * 4u;
    r->bytes[B_DXN] = r->bytes[B_Q] = r->bytes[B_DATT] = sm * E * 2u;
    r->bytes[B_K] = r->bytes[B_V] = sm * KV * 2u;
    r->bytes[B_DAWS] = pulsar_gqa_workspace_bytes(&B, (uint32_t)sm, s.max_seq);
    r->bytes[B_IDS] = (uint64_t)NP_PROMPT_MAX * 4u;
    r->bytes[B_LOGITS] = (uint64_t)s.vocab * 4u;
    r->bytes[B_POOLS] = (uint64_t)s.n_layer * r->pool_bytes;
    uint64_t total = 0;
    for (int b = 0; b < B_COUNT; b++) {
        r->t[b] = pulsar_gpu_tensor_alloc(r->bytes[b]);
        if (!r->t[b]) {
            fprintf(stderr, "pulsar: %s: the forward's scratch did not fit the device -- refusing\n", ARCH);
            return false;
        }
        total += r->bytes[b];
    }
    r->cls_rows = (const float *)tensor_data(&e->model, w->cls);
    fprintf(stderr, "pulsar: %s: forward scratch %.0f MiB (tower %u rows, decoder prefill %u rows, KV pools %.0f MiB, "
                    "decoder attention workspace %.0f MiB)\n", ARCH, (double)total / 1048576.0, r->n_max, r->s_max,
            (double)r->bytes[B_POOLS] / 1048576.0, (double)r->bytes[B_DAWS] / 1048576.0);
    return true;
}

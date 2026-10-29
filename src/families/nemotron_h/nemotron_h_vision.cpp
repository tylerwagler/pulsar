/* nemotron_h_vision.cpp -- L331: Nemotron 3.5 Super VL's vision tower on the image front (nemotron_h.h section 2b).
 *
 * What is the family's here is the composition; every block is shared:
 *   seed     res = [the 10 cls / register rows | the CPE position rows] (host f32, nemotron_h_image.cpp), then
 *            res[10 + i] += patch_i Wpatch  (the bf16 rows front door, f32 out; res32_add)
 *   tower    the shared ViT encoder (engine/vit_encoder.h, Parse's blocks) on bf16 operand rows: the BF16 rows arm and
 *            the bf16 FMHA
 *   project  xn = LN_final(res[10..])  (bf16 rows, merge-block order: four consecutive rows are one LLM row's pixel
 *            shuffle); p = RMSNorm_5120(xn)  (pulsar_rows_rmsnorm, the bf16 weight); h = relu^2(p W1) (bias_emit
 *            RELU2, bf16); y = h W2 (bf16 emit) -- the LLM rows, 4096 wide
 * THE FORMATS: bf16 operand rows into every Linear (the tower's weights are BF16, the projector's too), an f32
 * residual (RADIO's high-norm tokens: vit_encoder.h), f32 LayerNorm parameters; where pulsar rounds differently from
 * the checkpoint's fp32 reference it is graded (nemotron-vision-tower-gate, against tests/nemotron_vision_goldens.py's
 * fixture, the reference's own bf16-vs-fp32 floor).
 *
 * vision_final_layernorm (L331, plan 121 R1): the checkpoint's own code applies it -- modeling_nemotron_h_omni.py
 * 187-200 builds it whenever the language model has MTP (num_nextn_predict_layers 1 here: Megatron put a final
 * LayerNorm in every TransformerBlock built from the shared config, the vision tower's included), 222-223 applies it
 * before the pixel shuffle -- and vLLM does the same today (nano_nemotron_vl.py 1004-1012, 1108-1109, 1142-1143:
 * "Match the HF vision projector for checkpoints trained with MTP").  Its stored weights moved off their init
 * (weight 0.988 .. 1.016, bias up to 0.0195): they trained.  So it is the one path; the arms differ by 25% relative
 * RMS (the fixture's case 0), which is why the choice is not a rounding detail. */
#include "nemotron_h.h"
#include "cuda/pulsar_cuda_qwen.h"   /* the rows front door (family-neutral) */
#include "lib/bpe_tokenizer.h"

#include <math.h>
#include <stddef.h>
#include <time.h>

const pulsar_nemotron_h_vision_shape PULSAR_NEMOTRON_H_VISION_SHAPE = {
    /* vit  */ 1280, PULSAR_NEMOTRON_H_VIT_DEPTH, 16, 80, 5120, 16, 10, 128, 1e-6f,
    /* proj */ 2, 20480, 1e-5f,
    /* tile */ 1024, 13312,
    /* mean */ {0.48145466f, 0.4578275f, 0.40821073f},
    /* std  */ {0.26862954f, 0.26130258f, 0.27577711f},
};

#define NHV "nemotron_h vision: "

namespace {

const char *const OWNER = "nemotron_h vision";
constexpr uint32_t CHUNK = 2048;       /* the tower's Linear row chunk */
constexpr uint32_t PROJ_CHUNK = 512;   /* the projector's (its fc1 stage is 20480 wide) */
constexpr uint32_t LLM_WIDTH = 4096;

enum vk { K_U32, K_F32 };
struct vkey {
    const char *key;
    vk kind;
    size_t off;
};
#define U(k, f) {"nemotron_h_vision." k, K_U32, offsetof(pulsar_nemotron_h_vision_shape, f)}
#define F(k, f) {"nemotron_h_vision." k, K_F32, offsetof(pulsar_nemotron_h_vision_shape, f)}
const vkey k_keys[] = {
    U("vit.width", width), U("vit.depth", depth), U("vit.heads", heads), U("vit.head_dim", head_dim),
    U("vit.mlp", mlp), U("vit.patch", patch), U("vit.prefix", prefix), U("vit.pos_grid", pos_grid),
    F("vit.norm_eps", eps), U("proj.merge", merge), U("proj.hidden", proj_hidden), F("proj.norm_eps", proj_eps),
    U("image.min_patches", min_patches), U("image.max_patches", max_patches),
    F("image.mean.0", mean[0]), F("image.mean.1", mean[1]), F("image.mean.2", mean[2]),
    F("image.std.0", std[0]), F("image.std.1", std[1]), F("image.std.2", std[2]),
};
#undef U
#undef F

bool refuse(const char *fmt, const char *key) {
    fprintf(stderr, "pulsar: " NHV);
    fprintf(stderr, fmt, key);
    fprintf(stderr, " -- refusing\n");
    pulsar_load_refuse();
    return false;
}

bool read_shape(const pulsar_model *m, pulsar_nemotron_h_vision_shape *s) {
    *s = PULSAR_NEMOTRON_H_VISION_SHAPE;
    bool ok = true;
    for (uint64_t i = 0; i < m->n_kv; i++) {
        const pulsar_str k = m->kv[i].key;
        if (k.len < 18 || memcmp(k.ptr, "nemotron_h_vision.", 18) != 0) continue;
        bool known = false;
        for (const vkey &d : k_keys) known |= pulsar_streq(k, d.key);
        if (!known) {
            char key[160];
            snprintf(key, sizeof key, "%.*s", (int)(k.len < 150 ? k.len : 150), k.ptr);
            ok = refuse("%s is a tower fact this declaration does not know", key);
        }
    }
    for (const vkey &d : k_keys) {
        char *at = (char *)s + d.off;
        if (d.kind == K_U32) {
            uint32_t v = 0;
            if (!model_get_u32(m, d.key, &v)) ok = refuse("required tower fact %s is missing", d.key);
            else if (v != *(uint32_t *)at) ok = refuse("%s is not the profile's (C-RADIO ViT-H/16 + mlp1)", d.key);
        } else {
            float v = 0.0f;
            if (!model_get_f32_compat(m, d.key, &v)) ok = refuse("required tower fact %s is missing", d.key);
            else if (v != *(float *)at) ok = refuse("%s is not the profile's (C-RADIO ViT-H/16 + mlp1)", d.key);
        }
    }
    if (ok && (s->head_dim * s->heads != s->width || s->merge != 2 || s->width * s->merge * s->merge != 5120u))
        ok = refuse("%s: the profile's geometry does not compose", "nemotron_h_vision");
    return ok;
}

const pulsar_role_admit A_BF16 = {PULSAR_ROLE_PLAIN(PULSAR_TENSOR_BF16), false, PULSAR_ROLE_DENSE, 0};
const pulsar_role_admit A_F32 = {PULSAR_ROLE_PLAIN(PULSAR_TENSOR_F32), false, PULSAR_ROLE_DENSE, 0};
const pulsar_role_admit A_LINEAR = {0, true, PULSAR_ROLE_DENSE, PULSAR_ACTS(PULSAR_ACT_ROWS_BF16)};

float bf16f(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

}  // namespace

bool pulsar_nemotron_h_vision_bind(const pulsar_model *m, pulsar_role_binder *b, pulsar_nemotron_h_vision *v) {
    v->present = false;
    if (!read_shape(m, &v->shape)) return false;
    const pulsar_nemotron_h_vision_shape &s = v->shape;
    const uint64_t D = s.width, M = s.mlp, P = 3ull * s.patch * s.patch, G = (uint64_t)s.pos_grid * s.pos_grid;
    const uint64_t S = D * s.merge * s.merge;
    auto top = [](const char *r) { return pulsar_role_top(r); };
    v->patch = pulsar_role_bind(b, top("vit_patch"), A_LINEAR, 2, P, D);
    v->cls = pulsar_role_bind(b, top("vit_cls"), A_BF16, 2, D, s.prefix);
    v->pos = pulsar_role_bind(b, top("vit_pos"), A_BF16, 2, D, G);
    for (uint32_t il = 0; il < s.depth; il++) {
        auto k = [&](const char *r) { return pulsar_role_in(r, "vit", (int32_t)il); };
        pulsar_vit_block &B = v->vit[il];
        B.norm1 = pulsar_role_bind(b, k("vit_norm1"), A_F32, 1, D);
        B.norm1_b = pulsar_role_bind(b, k("vit_norm1_bias"), A_F32, 1, D);
        B.qkv = pulsar_role_bind(b, k("vit_qkv"), A_LINEAR, 2, D, 3u * D);
        B.qkv_b = pulsar_role_bind(b, k("vit_qkv_bias"), A_F32, 1, 3u * D);
        B.proj = pulsar_role_bind(b, k("vit_proj"), A_LINEAR, 2, D, D);
        B.proj_b = pulsar_role_bind(b, k("vit_proj_bias"), A_F32, 1, D);
        B.norm2 = pulsar_role_bind(b, k("vit_norm2"), A_F32, 1, D);
        B.norm2_b = pulsar_role_bind(b, k("vit_norm2_bias"), A_F32, 1, D);
        B.fc1 = pulsar_role_bind(b, k("vit_fc1"), A_LINEAR, 2, D, M);
        B.fc1_b = pulsar_role_bind(b, k("vit_fc1_bias"), A_F32, 1, M);
        B.fc2 = pulsar_role_bind(b, k("vit_fc2"), A_LINEAR, 2, M, D);
        B.fc2_b = pulsar_role_bind(b, k("vit_fc2_bias"), A_F32, 1, D);
    }
    v->final_norm = pulsar_role_bind(b, top("vit_final_norm"), A_F32, 1, D);
    v->final_norm_b = pulsar_role_bind(b, top("vit_final_norm_bias"), A_F32, 1, D);
    v->proj_norm = pulsar_role_bind(b, top("proj_norm"), A_BF16, 1, S);
    v->proj_fc1 = pulsar_role_bind(b, top("proj_fc1"), A_LINEAR, 2, S, s.proj_hidden);
    v->proj_fc2 = pulsar_role_bind(b, top("proj_fc2"), A_LINEAR, 2, s.proj_hidden, LLM_WIDTH);
    if (pulsar_load_refusals()) return false;
    /* the host tables: the position field and the prefix rows, widened from the mapping's bf16 */
    const uint16_t *pos = (const uint16_t *)tensor_data(m, v->pos), *cls = (const uint16_t *)tensor_data(m, v->cls);
    v->pos_field.resize(G * D);
    for (size_t i = 0; i < v->pos_field.size(); i++) v->pos_field[i] = bf16f(pos[i]);
    v->cls_rows.resize((size_t)s.prefix * D);
    for (size_t i = 0; i < v->cls_rows.size(); i++) v->cls_rows[i] = bf16f(cls[i]);
    v->present = true;
    return true;
}

bool pulsar_nemotron_h_vision_tokens(pulsar_nemotron_h_vision *v, const bpe_tokenizer *tok) {
    v->image_id = bpe_tokenizer_added_id(tok, "<image>");
    v->start_id = bpe_tokenizer_added_id(tok, "<img>");
    v->end_id = bpe_tokenizer_added_id(tok, "</img>");
    if (v->image_id < 0 || v->start_id < 0 || v->end_id < 0) {
        fprintf(stderr, "pulsar: " NHV "the tokenizer lacks <image> / <img> / </img> (%d / %d / %d) -- refusing\n",
                v->image_id, v->start_id, v->end_id);
        return false;
    }
    return true;
}

/* ---- the forward ---------------------------------------------------------------------------------------------- */

enum vbuf { V_PATCH, V_RES, V_XN, V_QKV, V_ATT, V_MLP, V_STAGE, V_LWS, V_COUNT };

struct pulsar_nemotron_h_vision_run {
    pulsar_gpu_tensor *t[V_COUNT];
    uint64_t bytes[V_COUNT];
    uint32_t n_max;   ///< prefix + the largest tile's patches
};

void pulsar_nemotron_h_vision_run_free(pulsar_nemotron_h_vision *v) {
    if (!v || !v->run) return;
    for (pulsar_gpu_tensor *t : v->run->t) pulsar_gpu_tensor_free(t);
    delete v->run;
    v->run = NULL;
}

bool pulsar_nemotron_h_vision_run_create(const pulsar_model *m, pulsar_nemotron_h_vision *v) {
    if (!v->present || v->run) return true;
    const pulsar_nemotron_h_vision_shape &s = v->shape;
    pulsar_nemotron_h_vision_run *r = new pulsar_nemotron_h_vision_run();
    v->run = r;   /* unload frees it, whatever this verdict */
    const uint64_t D = s.width, P = 3ull * s.patch * s.patch, n = s.prefix + s.max_patches;
    const uint64_t S = D * s.merge * s.merge;
    r->n_max = (uint32_t)n;
    /* the rows front door's workspace: the largest of the tower's and the projector's linears at their chunks */
    uint64_t lws = 0;
    {
        const struct { const pulsar_tensor *t; int in, out, rows; } ls[] = {
            {v->patch, (int)P, (int)D, (int)CHUNK}, {v->vit[0].qkv, (int)D, (int)(3 * D), (int)CHUNK},
            {v->vit[0].fc1, (int)D, (int)s.mlp, (int)CHUNK}, {v->vit[0].fc2, (int)s.mlp, (int)D, (int)CHUNK},
            {v->proj_fc1, (int)S, (int)s.proj_hidden, (int)PROJ_CHUNK},
            {v->proj_fc2, (int)s.proj_hidden, (int)LLM_WIDTH, (int)PROJ_CHUNK}};
        for (const auto &x : ls) {
            pulsar_rows_linear l;
            if (!pulsar_linear_rows_ref(m, x.t, x.in, x.out, true, "nemotron_h vision", &l)) return false;
            const uint64_t b = pulsar_rows_linear_workspace_bytes(&l, x.rows);
            if (b > lws) lws = b;
        }
    }
    r->bytes[V_PATCH] = (uint64_t)s.max_patches * P * 2u;
    r->bytes[V_RES] = n * D * 4u;
    r->bytes[V_XN] = n * D * 2u;            /* also the final LayerNorm's rows: [n_llm][S] */
    r->bytes[V_QKV] = n * 3u * D * 2u;      /* also the LLM rows: [n_llm][4096] */
    r->bytes[V_ATT] = n * D * 2u;           /* also the projector's RMSNorm rows: [n_llm][S] */
    r->bytes[V_MLP] = (uint64_t)PROJ_CHUNK * s.proj_hidden * 2u;
    r->bytes[V_STAGE] = (uint64_t)PROJ_CHUNK * s.proj_hidden * 4u;
    r->bytes[V_LWS] = lws ? lws : 16u;
    static_assert(CHUNK * 5120u <= PROJ_CHUNK * 20480u, "the tower's stage fits the projector's");
    if ((uint64_t)CHUNK * s.mlp > (uint64_t)PROJ_CHUNK * s.proj_hidden ||
        (uint64_t)(s.max_patches / 4u) * LLM_WIDTH > n * 3u * D) {
        fprintf(stderr, "pulsar: " NHV "the scratch plan does not cover the profile -- refusing\n");
        return false;
    }
    uint64_t total = 0;
    for (int b = 0; b < V_COUNT; b++) {
        r->t[b] = pulsar_gpu_tensor_alloc(r->bytes[b]);
        if (!r->t[b]) {
            fprintf(stderr, "pulsar: " NHV "the tower's scratch did not fit the device -- refusing\n");
            return false;
        }
        total += r->bytes[b];
    }
    fprintf(stderr, "pulsar: " NHV "C-RADIO ViT-H/16 (%u blocks of %u, bf16 rows, the shared ViT encoder + bf16 FMHA) "
                    "+ vision_final_layernorm + pixel shuffle + mlp1 -> %u; scratch %.0f MiB for %u rows\n",
            s.depth, s.width, LLM_WIDTH, (double)total / 1048576.0, r->n_max);
    return true;
}

namespace {

template <class T = void>
T *dp(const pulsar_gpu_tensor *t) { return (T *)pulsar_gpu_tensor_device_ptr(t); }

bool vfail(char *err, size_t errlen, const char *what) {
    if (err) snprintf(err, errlen, NHV "%s", what);
    return false;
}

/* stage [rows][out] f32 = x (bf16 rows) times t */
bool lin(const pulsar_model *m, const pulsar_nemotron_h_vision_run *r, const pulsar_tensor *t, int in, int out,
         const uint16_t *x, uint32_t rows, const char *what) {
    pulsar_rows_linear l;
    return pulsar_linear_rows_ref(m, t, in, out, true, what, &l) &&
           pulsar_rows_linear_launch(&l, x, (int)rows, dp<float>(r->t[V_STAGE]), dp(r->t[V_LWS]), r->bytes[V_LWS], 0) ==
               0;
}

double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1e3 * (double)ts.tv_sec + 1e-6 * (double)ts.tv_nsec;
}

}  // namespace

bool pulsar_nemotron_h_vision_encode(const pulsar_model *m, const pulsar_nemotron_h_vision *v, const uint8_t *bytes,
                                     size_t len, uint16_t **rows_out, int *n_rows_out, std::vector<float> *dbg,
                                     char *err, size_t errlen) {
    *rows_out = NULL;
    *n_rows_out = 0;
    const pulsar_nemotron_h_vision_run *r = v->run;
    if (!v->present || !r) return vfail(err, errlen, "no tower is bound");
    const pulsar_nemotron_h_vision_shape &s = v->shape;
    const double t0 = now_ms();
    pulsar_nemotron_h_image img;
    if (!pulsar_nemotron_h_image_prepare(&s, bytes, len, &img, err, errlen)) return false;
    const uint32_t np = img.wp * img.hp, NC = s.prefix, N = NC + np, D = s.width, P = 3u * s.patch * s.patch;
    const uint32_t S = D * s.merge * s.merge, n_llm = np / (s.merge * s.merge), H = s.proj_hidden;
    if (N > r->n_max) return vfail(err, errlen, "the tile is larger than the scratch");
    std::vector<float> pos;
    pulsar_nemotron_h_positions(&s, v->pos_field, img.wp, img.hp, &pos);
    float *res = dp<float>(r->t[V_RES]), *stage = dp<float>(r->t[V_STAGE]);
    uint16_t *patch = dp<uint16_t>(r->t[V_PATCH]), *xn = dp<uint16_t>(r->t[V_XN]), *qkv = dp<uint16_t>(r->t[V_QKV]),
             *att = dp<uint16_t>(r->t[V_ATT]), *mlp = dp<uint16_t>(r->t[V_MLP]);
    /* the seed: the prefix rows, the positions, then + the patch embedding */
    if (!pulsar_gpu_tensor_write(r->t[V_RES], 0, v->cls_rows.data(), (uint64_t)NC * D * 4u) ||
        !pulsar_gpu_tensor_write(r->t[V_RES], (uint64_t)NC * D * 4u, pos.data(), (uint64_t)np * D * 4u) ||
        !pulsar_gpu_tensor_write(r->t[V_PATCH], 0, img.patches.data(), (uint64_t)np * P * 2u))
        return vfail(err, errlen, "the image's rows did not reach the device");
    for (uint32_t r0 = 0; r0 < np; r0 += CHUNK) {
        const uint32_t k = np - r0 < CHUNK ? np - r0 : CHUNK;
        if (!lin(m, r, v->patch, (int)P, (int)D, patch + (size_t)r0 * P, k, "vision patch") ||
            pulsar_rows_res32_add_launch(res + (size_t)(NC + r0) * D, D, stage, D, NULL, k, D, NULL, NULL, 0.f, NULL, 0,
                                         PULSAR_ROWS_OUT_BF16, 0) != 0)
            return vfail(err, errlen, "the patch embedding refused");
    }
    const pulsar_vit_dims dims = {D, s.heads, s.head_dim, s.mlp, s.depth, s.eps};
    const pulsar_vit_bufs bufs = {res, xn, qkv, att, mlp, stage, dp(r->t[V_LWS]), r->bytes[V_LWS], CHUNK};
    if (!pulsar_vit_encode(m, dims, v->vit, PULSAR_ROWS_OUT_BF16, bufs, N, NULL, err, errlen)) return false;
    if (dbg) {   /* the tower's residual, the patches back in raster order */
        std::vector<float> all((size_t)N * D);
        if (!pulsar_gpu_tensor_read(r->t[V_RES], 0, all.data(), all.size() * 4u))
            return vfail(err, errlen, "the stage read failed");
        dbg->assign(all.begin(), all.begin() + (size_t)NC * D);
        dbg->resize(all.size());
        for (uint32_t pr = 0; pr < img.hp; pr++)
            for (uint32_t pc = 0; pc < img.wp; pc++)
                memcpy(dbg->data() + (size_t)(NC + pr * img.wp + pc) * D,
                       all.data() + (size_t)(NC + pulsar_nemotron_h_patch_slot(pr, pc, img.wp)) * D, (size_t)D * 4u);
    }
    /* the projector: the final LayerNorm on the patches (merge-block rows: four make an LLM row), mlp1 */
    const float *fw = (const float *)pulsar_weight_device_ptr(m, v->final_norm, "vision final norm");
    const float *fb = (const float *)pulsar_weight_device_ptr(m, v->final_norm_b, "vision final norm");
    const uint16_t *pw = (const uint16_t *)pulsar_weight_device_ptr(m, v->proj_norm, "vision proj norm");
    if (pulsar_rows_layernorm32_launch(res + (size_t)NC * D, true, D, NULL, np, D, fw, fb, s.eps, xn, D,
                                       PULSAR_ROWS_OUT_BF16, NULL, 0, 0) != 0 ||
        pulsar_rows_rmsnorm_launch(xn, S, n_llm, S, pw, s.proj_eps, att, S, 0) != 0)
        return vfail(err, errlen, "the projector's norms refused");
    for (uint32_t r0 = 0; r0 < n_llm; r0 += PROJ_CHUNK) {
        const uint32_t k = n_llm - r0 < PROJ_CHUNK ? n_llm - r0 : PROJ_CHUNK;
        if (!lin(m, r, v->proj_fc1, (int)S, (int)H, att + (size_t)r0 * S, k, "vision mlp1 fc1") ||
            pulsar_rows_bias_emit_launch(stage, H, k, H, NULL, PULSAR_ROWS_ACT_RELU2, mlp, H, PULSAR_ROWS_OUT_BF16, 0) !=
                0 ||
            !lin(m, r, v->proj_fc2, (int)H, (int)LLM_WIDTH, mlp, k, "vision mlp1 fc2") ||
            pulsar_rows_bias_emit_launch(stage, LLM_WIDTH, k, LLM_WIDTH, NULL, PULSAR_ROWS_ACT_NONE,
                                         qkv + (size_t)r0 * LLM_WIDTH, LLM_WIDTH, PULSAR_ROWS_OUT_BF16, 0) != 0)
            return vfail(err, errlen, "the projector's mlp1 refused");
    }
    uint16_t *out = (uint16_t *)malloc((size_t)n_llm * LLM_WIDTH * 2u);
    if (!out || !pulsar_gpu_tensor_read(r->t[V_QKV], 0, out, (uint64_t)n_llm * LLM_WIDTH * 2u)) {
        free(out);
        return vfail(err, errlen, "the rows' read failed");
    }
    /* rule 5: the lane taken, per image */
    fprintf(stderr, "pulsar: " NHV "tile %ux%u patches -> %u rows (tower + final norm + mlp1, bf16 rows) %.1f ms\n",
            img.wp, img.hp, n_llm, now_ms() - t0);
    *rows_out = out;
    *n_rows_out = (int)n_llm;
    return true;
}

/* ---- the image front (family.h pulsar_family_vision) ------------------------------------------------------------
 *
 * The renderer writes `<img><image></img>` where an image goes (the processor's expansion of the template's `<image>`:
 * a text substitution before tokenisation, so the ids are the same); the expansion replaces that `<image>` with the
 * tile's (w_p h_p / 4) `<image>` ids, the BLOCK, whose rows are the projector's (row-major over the merged grid). */

namespace {

const pulsar_nemotron_h_vision *vision_of(const pulsar_engine *e) {
    const pulsar_nemotron_h_model *w = pulsar_nemotron_h_of(e);
    return w && w->vision.present ? &w->vision : NULL;
}

int32_t image_id(const pulsar_engine *e) {
    const pulsar_nemotron_h_vision *v = vision_of(e);
    return v ? v->image_id : -1;
}

bool is_sentinel(const pulsar_engine *e, int32_t id) { return image_id(e) >= 0 && id == image_id(e); }

bool block_extent(const pulsar_engine *e, const int32_t *ids, int n, int start, int *len) {
    const int32_t t = image_id(e);
    if (t < 0 || !ids || start < 0 || start >= n || ids[start] != t || (start > 0 && ids[start - 1] == t)) return false;
    int k = start;
    while (k < n && ids[k] == t) k++;
    *len = k - start;
    return true;
}

int placeholder_id(const pulsar_engine *e) { return image_id(e); }

/* the tile of an image, decoded once per image per process (keyed by content: the expansion and the licence ask) */
struct tile_entry { uint64_t key; size_t len; uint32_t wp, hp; };
tile_entry g_tile[16];
int g_tile_n = 0, g_tile_next = 0;

bool tile_of(const pulsar_nemotron_h_vision *v, const pulsar_image_ref *img, uint32_t *wp, uint32_t *hp) {
    const uint64_t key = pulsar_image_content_hash(img);
    for (int i = 0; i < g_tile_n; i++)
        if (g_tile[i].key == key && g_tile[i].len == img->len) {
            *wp = g_tile[i].wp;
            *hp = g_tile[i].hp;
            return true;
        }
    uint8_t *rgb = NULL;
    int w = 0, h = 0;
    if (!img->bytes || !vision_decode_rgb(img->bytes, img->len, &rgb, &w, &h)) return false;
    free(rgb);
    pulsar_nemotron_h_tile(&v->shape, w, h, wp, hp);
    if (!*wp || !*hp || *wp * *hp > v->shape.max_patches) return false;
    const int slot = g_tile_n < 16 ? g_tile_n++ : (g_tile_next++ % 16);
    g_tile[slot] = {key, img->len, *wp, *hp};
    return true;
}

bool expand(const pulsar_engine *e, const pulsar_image_ref *img, pulsar_tokens *out, char *err, size_t errlen) {
    const pulsar_nemotron_h_vision *v = vision_of(e);
    uint32_t wp = 0, hp = 0;
    if (!v || !tile_of(v, img, &wp, &hp)) {
        snprintf(err, errlen, "an image could not be decoded or is not one the vision tower accepts");
        return false;
    }
    const uint32_t n = wp * hp / (v->shape.merge * v->shape.merge);
    for (uint32_t i = 0; i < n; i++) pulsar_tokens_push(out, v->image_id);
    return true;
}

uint32_t row_width(const pulsar_engine *) { return LLM_WIDTH; }

bool block_rows(const pulsar_engine *e, const pulsar_image_ref *img, const int32_t *, int block_len,
                const uint16_t *tower, int n_tower, uint16_t **tower_out, int *n_tower_out, uint16_t *out, char *err,
                size_t errlen) {
    const pulsar_nemotron_h_vision *v = vision_of(e);
    if (!v) {
        snprintf(err, errlen, "no vision tower is bound");
        return false;
    }
    if (!tower) {
        uint16_t *rows = NULL;
        int n = 0;
        if (!pulsar_nemotron_h_vision_encode(&e->model, v, img->bytes, img->len, &rows, &n, NULL, err, errlen))
            return false;
        *tower_out = rows;
        *n_tower_out = n;
        tower = rows;
        n_tower = n;
    }
    if (n_tower != block_len) {
        snprintf(err, errlen, "the tower made %d rows for a %d-token block", n_tower, block_len);
        return false;
    }
    memcpy(out, tower, (size_t)block_len * LLM_WIDTH * sizeof(uint16_t));
    return true;
}

}  // namespace

const pulsar_family_vision PULSAR_NEMOTRON_H_IMAGE_FRONT = {
    /* .is_sentinel      = */ is_sentinel,
    /* .block_extent     = */ block_extent,
    /* .placeholder_text = */ "<img><image></img>",
    /* .placeholder_id   = */ placeholder_id,
    /* .expand           = */ expand,
    /* .row_width        = */ row_width,
    /* .block_rows       = */ block_rows,
    /* .grid             = */ NULL,   /* NoPE: a block has no positions to lay out */
};

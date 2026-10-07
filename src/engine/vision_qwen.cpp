/* vision_qwen.cpp -- L268: Qwen3.8-Flash-Next's own part of the image path.
 *
 * Everything an image shares with DeepSeek's lives elsewhere once: the codec and Pillow's resize (vision.cpp), the
 * placeholder walk, block fit, licence, chunk merge and tower-output cache (image_front.cpp), the identity a session
 * holds (image_identity.cpp).  What is here is Qwen's:
 *
 *   - preprocessing (HF Qwen2VLImageProcessorPil, the checkpoint's preprocessor_config.json): smart_resize to a
 *     multiple of patch x merge (32) inside [min_pixels, max_pixels], Pillow BICUBIC, rescale 1/255 in float64 then
 *     float32, normalise with mean = std = 0.5 in float32, patchify -- rows in 2x2 merge-block order (block row,
 *     block col, row in block, col in block), columns (channel, frame, y, x) with the still image's frame repeated
 *     (temporal_patch_size 2).
 *
 * Graded by vision-qwen-pixel-gate against HF's own processor (tests/vision_qwen_pixel_goldens.py). */
#include "pulsar_engine_internal.h"
#include "family_qwen.h"

#include <math.h>

bool qwen_vision_smart_resize(int h, int w, int *h_out, int *w_out, char *err, size_t errlen) {
    const int factor = (int)(PULSAR_QWEN_VISION_PATCH * PULSAR_QWEN_VISION_MERGE);
    if (h <= 0 || w <= 0) {
        snprintf(err, errlen, "image size %dx%d", w, h);
        return false;
    }
    if ((double)(h > w ? h : w) / (double)(h < w ? h : w) > 200.0) {
        snprintf(err, errlen, "image aspect ratio %dx%d is past 200", w, h);
        return false;
    }
    /* Python's round() is half to even: nearbyint under the default rounding mode */
    long hb = (long)nearbyint((double)h / factor) * factor;
    long wb = (long)nearbyint((double)w / factor) * factor;
    const double hw = (double)h * (double)w;
    if ((double)hb * (double)wb > (double)PULSAR_QWEN_VISION_MAX_PIXELS) {
        const double beta = sqrt(hw / (double)PULSAR_QWEN_VISION_MAX_PIXELS);
        hb = (long)floor((double)h / beta / factor) * factor;
        wb = (long)floor((double)w / beta / factor) * factor;
        if (hb < factor) hb = factor;
        if (wb < factor) wb = factor;
    } else if ((double)hb * (double)wb < (double)PULSAR_QWEN_VISION_MIN_PIXELS) {
        const double beta = sqrt((double)PULSAR_QWEN_VISION_MIN_PIXELS / hw);
        hb = (long)ceil((double)h * beta / factor) * factor;
        wb = (long)ceil((double)w * beta / factor) * factor;
    }
    *h_out = (int)hb;
    *w_out = (int)wb;
    return true;
}

void qwen_vision_pixels_free(qwen_vision_pixels *p) {
    if (!p) return;
    free(p->values);
    memset(p, 0, sizeof *p);
}

bool qwen_vision_preprocess(const uint8_t *bytes, size_t len, qwen_vision_pixels *out, char *err, size_t errlen) {
    memset(out, 0, sizeof *out);
    uint8_t *rgb = NULL;
    int w = 0, h = 0;
    if (!vision_decode_rgb(bytes, len, &rgb, &w, &h)) {
        snprintf(err, errlen, "the image could not be decoded (PNG or JPEG)");
        return false;
    }
    int rh = 0, rw = 0;
    if (!qwen_vision_smart_resize(h, w, &rh, &rw, err, errlen)) {
        free(rgb);
        return false;
    }
    uint8_t *px = vision_pil_resize_rgb(rgb, w, h, rw, rh);
    free(rgb);
    if (!px) {
        snprintf(err, errlen, "resizing %dx%d to %dx%d failed", w, h, rw, rh);
        return false;
    }
    const int P = (int)PULSAR_QWEN_VISION_PATCH, M = (int)PULSAR_QWEN_VISION_MERGE,
              T = (int)PULSAR_QWEN_VISION_TEMPORAL;
    const int gh = rh / P, gw = rw / P;
    const int cols = 3 * T * P * P;
    out->values = (float *)malloc((size_t)gh * gw * cols * sizeof(float));
    if (!out->values) {
        free(px);
        snprintf(err, errlen, "out of memory for %dx%d patches", gh, gw);
        return false;
    }
    /* rescale: float64(u8) * (1/255), then float32; normalise: (x - 0.5f) / 0.5f in float32 */
    float lut[256];
    for (int v = 0; v < 256; v++) {
        const float r = (float)((double)v * (1.0 / 255.0));
        lut[v] = (r - (float)PULSAR_QWEN_VISION_MEAN) / (float)PULSAR_QWEN_VISION_STD;
    }
    int row = 0;
    for (int br = 0; br < gh / M; br++)
        for (int bc = 0; bc < gw / M; bc++)
            for (int ir = 0; ir < M; ir++)
                for (int ic = 0; ic < M; ic++, row++) {
                    const int py0 = (br * M + ir) * P, px0 = (bc * M + ic) * P;
                    float *dst = out->values + (size_t)row * cols;
                    for (int c = 0; c < 3; c++)
                        for (int t = 0; t < T; t++)
                            for (int y = 0; y < P; y++)
                                for (int x = 0; x < P; x++)
                                    dst[((c * T + t) * P + y) * P + x] =
                                        lut[px[((size_t)(py0 + y) * rw + (px0 + x)) * 3 + c]];
                }
    free(px);
    out->rows = gh * gw;
    out->cols = cols;
    out->grid_h = gh;
    out->grid_w = gw;
    out->resized_h = rh;
    out->resized_w = rw;
    return true;
}

/* ---- the tower (L268 S3) ------------------------------------------------------------------------------------- */

/* A bf16 tower tensor with exactly these dims (fastest-varying first), as a pointer into the mapping. */
static const void *qv_bind(const pulsar_model *m, bool *ok, const char *name, uint32_t nd, const uint64_t *dims) {
    const pulsar_tensor *t = pulsar_tensor_bind(m, PULSAR_QWEN_ARCH, name);
    if (!t) { *ok = false; return NULL; }
    bool good = t->type == PULSAR_TENSOR_BF16 && t->ndim == nd;
    for (uint32_t i = 0; good && i < nd; i++) good = t->dim[i] == dims[i];
    if (!good) {
        fprintf(stderr, "pulsar: %s: vision tensor %s is not bf16 of the tower's shape\n", PULSAR_QWEN_ARCH, name);
        *ok = false;
        return NULL;
    }
    return (const char *)tensor_map_base(m, t) + t->abs_offset;
}

bool qwen_vision_bind(const pulsar_model *m, pulsar_qwen_vision_weights_dev *w, bool *present) {
    *present = model_find_tensor(m, "model.visual.patch_embed.proj.weight") != NULL;
    if (!*present) return true;   /* a text-only artifact: images are refused by name */
    const uint64_t D = PULSAR_QWEN_VISION_DIM, I = PULSAR_QWEN_VISION_INTER, O = PULSAR_QWEN_VISION_OUT,
                   P = PULSAR_QWEN_VISION_PATCH, T = PULSAR_QWEN_VISION_TEMPORAL, M = 4 * D,
                   S = (uint64_t)PULSAR_QWEN_VISION_POS_SIDE * PULSAR_QWEN_VISION_POS_SIDE;
    bool ok = true;
    char name[160];
    {
        const uint64_t d[5] = {P, P, T, 3, D};
        w->patch_w = qv_bind(m, &ok, "model.visual.patch_embed.proj.weight", 5, d);
    }
#define QV1(field, nm, n0) do { const uint64_t d_[1] = {(n0)}; (field) = qv_bind(m, &ok, (nm), 1, d_); } while (0)
#define QV2(field, nm, n0, n1) do { const uint64_t d_[2] = {(n0), (n1)}; (field) = qv_bind(m, &ok, (nm), 2, d_); } while (0)
    QV1(w->patch_b, "model.visual.patch_embed.proj.bias", D);
    QV2(w->pos_embed, "model.visual.pos_embed.weight", D, S);
    for (uint32_t l = 0; l < PULSAR_QWEN_VISION_LAYERS; l++) {
        pulsar_qwen_vision_block_dev *b = &w->block[l];
#define QVB1(field, suffix, n0) do { snprintf(name, sizeof name, "model.visual.blocks.%u.%s", l, (suffix)); QV1(field, name, n0); } while (0)
#define QVB2(field, suffix, n0, n1) do { snprintf(name, sizeof name, "model.visual.blocks.%u.%s", l, (suffix)); QV2(field, name, n0, n1); } while (0)
        QVB1(b->norm1_w, "norm1.weight", D);
        QVB1(b->norm1_b, "norm1.bias", D);
        QVB2(b->qkv_w, "attn.qkv.weight", D, 3 * D);
        QVB1(b->qkv_b, "attn.qkv.bias", 3 * D);
        QVB2(b->proj_w, "attn.proj.weight", D, D);
        QVB1(b->proj_b, "attn.proj.bias", D);
        QVB1(b->norm2_w, "norm2.weight", D);
        QVB1(b->norm2_b, "norm2.bias", D);
        QVB2(b->fc1_w, "mlp.linear_fc1.weight", D, I);
        QVB1(b->fc1_b, "mlp.linear_fc1.bias", I);
        QVB2(b->fc2_w, "mlp.linear_fc2.weight", I, D);
        QVB1(b->fc2_b, "mlp.linear_fc2.bias", D);
#undef QVB1
#undef QVB2
    }
    QV1(w->merger_norm_w, "model.visual.merger.norm.weight", D);
    QV1(w->merger_norm_b, "model.visual.merger.norm.bias", D);
    QV2(w->merger_fc1_w, "model.visual.merger.linear_fc1.weight", M, M);
    QV1(w->merger_fc1_b, "model.visual.merger.linear_fc1.bias", M);
    QV2(w->merger_fc2_w, "model.visual.merger.linear_fc2.weight", M, O);
    QV1(w->merger_fc2_b, "model.visual.merger.linear_fc2.bias", O);
#undef QV1
#undef QV2
    if (!ok) fprintf(stderr, "pulsar: %s: the artifact's vision tower does not bind -- refusing\n", PULSAR_QWEN_ARCH);
    return ok;
}

/* HF get_vision_interpolation_indices_and_weights (bilinear, align_corners) for one axis: the two taps into the
 * 48-wide table and their weights, in float32 as the reference computes them. */
static void qv_axis_taps(int index, int size, int *t0, int *t1, float *w0, float *w1) {
    const int side = (int)PULSAR_QWEN_VISION_POS_SIDE;
    const float src = (float)index * (float)(side - 1) / (float)(size - 1 > 1 ? size - 1 : 1);
    const float fl = floorf(src);
    const int base = (int)fl;
    *t0 = base < 0 ? 0 : base > side - 1 ? side - 1 : base;
    *t1 = base + 1 < 0 ? 0 : base + 1 > side - 1 ? side - 1 : base + 1;
    const float d0 = fabsf(src - fl - 0.0f), d1 = fabsf(src - fl - 1.0f);
    *w0 = 1.0f - d0 > 0.0f ? 1.0f - d0 : 0.0f;
    *w1 = 1.0f - d1 > 0.0f ? 1.0f - d1 : 0.0f;
}

/* float32 -> bfloat16, round to nearest even (torch's .to(bfloat16)). */
static uint16_t qv_f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

bool qwen_vision_encode(const pulsar_qwen_vision_weights_dev *w, const uint8_t *bytes, size_t len, uint16_t **rows,
                        int *n_rows, uint16_t **dbg, int *n_patches, char *err, size_t errlen) {
    *rows = NULL;
    *n_rows = 0;
    if (dbg) *dbg = NULL;
    qwen_vision_pixels px;
    if (!qwen_vision_preprocess(bytes, len, &px, err, errlen)) return false;
    const int n = px.rows, M = (int)PULSAR_QWEN_VISION_MERGE, gw = px.grid_w, gh = px.grid_h;
    const int n_out = n / (M * M);
    uint16_t *patches = (uint16_t *)malloc((size_t)n * px.cols * sizeof(uint16_t));
    int32_t *pos = (int32_t *)malloc((size_t)n * 2 * sizeof(int32_t));
    int32_t *idx = (int32_t *)malloc((size_t)n * 4 * sizeof(int32_t));
    float *wt = (float *)malloc((size_t)n * 4 * sizeof(float));
    uint16_t *out = (uint16_t *)malloc((size_t)n_out * PULSAR_QWEN_VISION_OUT * sizeof(uint16_t));
    uint16_t *d = dbg ? (uint16_t *)malloc((size_t)3 * n * PULSAR_QWEN_VISION_DIM * sizeof(uint16_t)) : NULL;
    bool ok = patches && pos && idx && wt && out && (!dbg || d);
    if (ok) {
        /* the model casts pixel_values to the tower's dtype */
        for (size_t i = 0; i < (size_t)n * px.cols; i++) patches[i] = qv_f32_to_bf16(px.values[i]);
        /* each patch's (row, col) from its merge-block position -- the rope's and the learned table's */
        const int side = (int)PULSAR_QWEN_VISION_POS_SIDE;
        for (int i = 0; i < n; i++) {
            const int in_col = i % M, in_row = (i / M) % M, block_col = (i / (M * M)) % (gw / M),
                      block_row = i / (M * M * (gw / M));
            const int r = block_row * M + in_row, c = block_col * M + in_col;
            pos[2 * i] = r;
            pos[2 * i + 1] = c;
            int h0, h1, w0, w1;
            float hw0, hw1, ww0, ww1;
            qv_axis_taps(r, gh, &h0, &h1, &hw0, &hw1);
            qv_axis_taps(c, gw, &w0, &w1, &ww0, &ww1);
            idx[4 * i + 0] = h0 * side + w0; wt[4 * i + 0] = hw0 * ww0;
            idx[4 * i + 1] = h0 * side + w1; wt[4 * i + 1] = hw0 * ww1;
            idx[4 * i + 2] = h1 * side + w0; wt[4 * i + 2] = hw1 * ww0;
            idx[4 * i + 3] = h1 * side + w1; wt[4 * i + 3] = hw1 * ww1;
        }
        ok = pulsar_cuda_qwen_vision_forward(w, patches, pos, idx, wt, n, out,
                                             n_out * (int)PULSAR_QWEN_VISION_OUT, d) != 0;
        if (!ok) snprintf(err, errlen, "the vision tower failed (%d patches)", n);
    } else {
        snprintf(err, errlen, "out of memory for %d patches", n);
    }
    free(patches); free(pos); free(idx); free(wt);
    if (n_patches) *n_patches = n;
    qwen_vision_pixels_free(&px);
    if (!ok) { free(out); free(d); return false; }
    *rows = out;
    *n_rows = n_out;
    if (dbg) *dbg = d;
    return true;
}

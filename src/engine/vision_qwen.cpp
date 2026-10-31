/** @file vision_qwen.cpp
 *  @brief L268: Qwen3.8-Flash-Next's own part of the image path. */
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

/// Free a pixel buffer's values and zero it.
/// @param p the buffer (NULL is a no-op)
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
    uint8_t *px = vision_pil_resize_rgb(rgb, w, h, rw, rh, PULSAR_PIL_BICUBIC);
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
/// @param m the model
/// @param[in,out] ok cleared on a refusal
/// @param name the tensor name
/// @param nd dims
/// @param dims [nd] the required dims
/// @return the mapped bytes, or NULL
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

/** @def QV1(field, nm, n0)
 *  qwen_vision_bind: bind 1-D bf16 tensor `nm` of [n0] into `field`. */
/** @def QV2(field, nm, n0, n1)
 *  qwen_vision_bind: bind 2-D bf16 tensor `nm` of [n0][n1] (fastest first) into `field`. */
/** @def QVB1(field, suffix, n0)
 *  qwen_vision_bind: QV1 on block l's tensor `model.visual.blocks.<l>.<suffix>`. */
/** @def QVB2(field, suffix, n0, n1)
 *  qwen_vision_bind: QV2 on block l's tensor `model.visual.blocks.<l>.<suffix>`. */
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
/// @param index the grid index on this axis
/// @param size the grid's extent on this axis
/// @param[out] t0 first tap
/// @param[out] t1 second tap
/// @param[out] w0 its weight
/// @param[out] w1 its weight
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
        for (size_t i = 0; i < (size_t)n * px.cols; i++) patches[i] = pulsar_bf16_from_f32(px.values[i]);
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

/* ---- Qwen's image front (L268; pulsar_family_vision, family.h) ------------------------------------------------
 *
 * A Qwen image is `<|vision_start|>` + (h' x w') `<|image_pad|>` + `<|vision_end|>` -- all in the vocabulary.  The
 * BLOCK is the run of pads (vision_start / vision_end are text: they embed as tokens and take text positions, as
 * HF's get_rope_index has them); the pads' rows are the merger's output, row-major over the merged grid
 * (h' = grid_h / 2, w' = grid_w / 2), which is also the grid their multi-axis rope positions follow. */

/// @param e the engine
/// @return the image pad id, or -1 without a tower
static int32_t qwen_pad_id(const pulsar_engine *e) {
    return e && e->qwen_weights ? e->qwen_weights->vision_pad_id : -1;
}

/// pulsar_family_vision::is_sentinel.
/// @param e the engine
/// @param id a token id
/// @return id is the pad id
static bool qwen_image_is_sentinel(const pulsar_engine *e, int32_t id) {
    const int32_t pad = qwen_pad_id(e);
    return pad >= 0 && id == pad;
}

/// pulsar_family_vision::block_extent.
/// @param e the engine
/// @param ids the sequence
/// @param n its length
/// @param start where to look
/// @param[out] len the run of pads at start
/// @return true when ids[start] is a pad
static bool qwen_image_block_extent(const pulsar_engine *e, const int32_t *ids, int n, int start, int *len) {
    const int32_t pad = qwen_pad_id(e);
    if (pad < 0 || !ids || start < 0 || start >= n || ids[start] != pad || (start > 0 && ids[start - 1] == pad))
        return false;
    int k = start;
    while (k < n && ids[k] == pad) k++;
    *len = k - start;
    return true;
}

/// pulsar_family_vision::placeholder_id.
/// @param e the engine
/// @return the pad id, or -1
static int qwen_image_placeholder_id(const pulsar_engine *e) { return qwen_pad_id(e); }

/* The merged grid of an image -- smart_resize's patch grid over the merge -- decoded once per image per process
 * (keyed by content: the licence, the identity and the expansion all ask). */
/** One cached merged grid. */
struct qwen_grid_entry { uint64_t key; size_t len; uint32_t h, w; };
/** @var qwen_grid_entry::key
 *  pulsar_image_content_hash of the image */
/** @var qwen_grid_entry::len
 *  the image's byte length, checked with the key */
/** @var qwen_grid_entry::h
 *  merged grid height: resized pixels / (patch x merge) */
/** @var qwen_grid_entry::w
 *  merged grid width: resized pixels / (patch x merge) */
/// The merged-grid cache, 16 entries, FIFO-replaced; process-global, unlocked.
static qwen_grid_entry g_qwen_grid[16];
/// Entries used (<= 16).
static int g_qwen_grid_n = 0, g_qwen_grid_next = 0;
                                                ///< replacements made once full (slot = next % 16)

/// pulsar_family_vision::grid.
/// @param img the image
/// @param[out] h merged grid height
/// @param[out] w merged grid width
/// @return false when it does not decode or resize
static bool qwen_image_grid(const pulsar_engine *, const pulsar_image_ref *img, uint32_t *h, uint32_t *w) {
    const uint64_t key = pulsar_image_content_hash(img);
    for (int i = 0; i < g_qwen_grid_n; i++)
        if (g_qwen_grid[i].key == key && g_qwen_grid[i].len == img->len) {
            *h = g_qwen_grid[i].h;
            *w = g_qwen_grid[i].w;
            return true;
        }
    uint8_t *rgb = NULL;
    int iw = 0, ih = 0;
    if (!img->bytes || !vision_decode_rgb(img->bytes, img->len, &rgb, &iw, &ih)) return false;
    free(rgb);
    int rh = 0, rw = 0;
    char err[128];
    if (!qwen_vision_smart_resize(ih, iw, &rh, &rw, err, sizeof err)) return false;
    const uint32_t f = PULSAR_QWEN_VISION_PATCH * PULSAR_QWEN_VISION_MERGE;
    *h = (uint32_t)rh / f;
    *w = (uint32_t)rw / f;
    const int slot = g_qwen_grid_n < 16 ? g_qwen_grid_n++ : (g_qwen_grid_next++ % 16);
    g_qwen_grid[slot].key = key;
    g_qwen_grid[slot].len = img->len;
    g_qwen_grid[slot].h = *h;
    g_qwen_grid[slot].w = *w;
    return true;
}

/// pulsar_family_vision::expand: h x w pads for one image.
/// @param e the engine
/// @param img the image
/// @param[in,out] out ids appended
/// @param err message buffer
/// @param errlen its size
/// @return false + err when it does not decode
static bool qwen_image_expand(const pulsar_engine *e, const pulsar_image_ref *img, pulsar_tokens *out, char *err,
                              size_t errlen) {
    uint32_t h = 0, w = 0;
    if (!qwen_image_grid(e, img, &h, &w)) {
        snprintf(err, errlen, "an image could not be decoded or is not one the vision tower accepts");
        return false;
    }
    for (uint32_t i = 0; i < h * w; i++) pulsar_tokens_push(out, qwen_pad_id(e));
    return true;
}

/// pulsar_family_vision::row_width.
/// @return PULSAR_QWEN_VISION_OUT
static uint32_t qwen_image_row_width(const pulsar_engine *) { return PULSAR_QWEN_VISION_OUT; }

/// pulsar_family_vision::block_rows: the block's rows, from the cached tower output or a fresh encode.
/// @param e the engine
/// @param img the image
/// @param block_len the block's pads
/// @param tower cached tower rows, or NULL
/// @param n_tower their count
/// @param[out] tower_out a fresh encode's rows (caller owns), when tower is NULL
/// @param[out] n_tower_out their count
/// @param[out] out [block_len][PULSAR_QWEN_VISION_OUT] bf16
/// @param err message buffer
/// @param errlen its size
/// @return false + err on a failed encode or a row count that is not the block's
static bool qwen_image_block_rows(const pulsar_engine *e, const pulsar_image_ref *img, const int32_t *, int block_len,
                                  const uint16_t *tower, int n_tower, uint16_t **tower_out, int *n_tower_out,
                                  uint16_t *out, char *err, size_t errlen) {
    if (!tower) {
        uint16_t *rows = NULL;
        int n = 0;
        if (!qwen_vision_encode(&e->qwen_weights->vision, img->bytes, img->len, &rows, &n, NULL, NULL, err, errlen))
            return false;
        *tower_out = rows;
        *n_tower_out = n;
        tower = rows;
        n_tower = n;
    }
    if (n_tower != block_len) {
        snprintf(err, errlen, "the tower made %d rows for a %d-pad block", n_tower, block_len);
        return false;
    }
    memcpy(out, tower, (size_t)block_len * PULSAR_QWEN_VISION_OUT * sizeof(uint16_t));
    return true;
}

const pulsar_family_vision PULSAR_QWEN_IMAGE_FRONT = {
    /* .is_sentinel      = */ qwen_image_is_sentinel,
    /* .block_extent     = */ qwen_image_block_extent,
    /* .placeholder_text = */ "<|vision_start|><|image_pad|><|vision_end|>",
    /* .placeholder_id   = */ qwen_image_placeholder_id,
    /* .expand           = */ qwen_image_expand,
    /* .row_width        = */ qwen_image_row_width,
    /* .block_rows       = */ qwen_image_block_rows,
    /* .grid             = */ qwen_image_grid,
};

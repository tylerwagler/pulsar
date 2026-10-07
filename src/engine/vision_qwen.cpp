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

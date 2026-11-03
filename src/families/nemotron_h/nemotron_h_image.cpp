/* nemotron_h_image.cpp -- L331: Nemotron 3.5 Super VL's image preprocessing and the tower's CPE positions, on the host
 * (nemotron_h.h section 2b).  Built WITHOUT -ffast-math (Makefile): the tile rule is the processor's float64 Python
 * and the pixels its float32 torch, evaluated as written.
 *
 * THE TILE (image_processing_nemotron_h_omni.py _compute_target_patches, at the per-image budget: max(min((16384 - 4)
 * x 4, max_patches), min_patches) = max_patches whatever the image count):
 *   ch = round(h / 16 + 0.5), cw = round(w / 16 + 0.5)          Python's round: half to even (so 16 px -> 2 patches)
 *   factor = min(sqrt(budget / (ch cw)), 1);  th = floor(factor ch), tw = floor(factor cw)
 *   below min_patches: up = sqrt(min / (th tw)), th = ceil(up th), tw = ceil(up tw)
 *   each side to a multiple of 2: up by one when the product stays within the budget, else down (at least 2)
 * THE PIXELS: the uint8 RGB widened to f32, torch's antialiased bicubic to (16 th, 16 tw) when the size differs
 * (vision_torch_bicubic_aa_rgb), (t / 255 - mean) / std in f32, bf16 (the projector casts pixel_values to the tower's
 * dtype).  THE PATCHES: modeling_radio's _image_to_patches -- element c * 256 + y * 16 + x of patch (r, col) is pixel
 * (16 r + y, 16 col + x) of channel c -- placed in merge-block order (pulsar_nemotron_h_patch_slot).
 * THE POSITIONS (modeling_radio _interpolate_position_embedding): the 128 x 128 table bilinear-resized (align_corners
 * False, no antialias; torch's channels-last CPU kernel: w00 i00 + w01 i01 + w10 i10 + w11 i11 with w = lambda_h
 * lambda_w) to M x M, M = max(th, tw), and its top-left th x tw (a square that covers the tile, cropped: never a
 * second resize), in f32 -- the residual's seed. */
#include "nemotron_h.h"

#include <math.h>
#include <stdlib.h>

void pulsar_nemotron_h_tile(const pulsar_nemotron_h_vision_shape *s, int w, int h, uint32_t *wp, uint32_t *hp) {
    const double P = (double)s->patch;
    const long budget = (long)s->max_patches, lo = (long)s->min_patches;
    const long ch = (long)nearbyint((double)h / P + 0.5), cw = (long)nearbyint((double)w / P + 0.5);
    const double f0 = sqrt((double)budget / (double)(ch * cw)), factor = f0 < 1.0 ? f0 : 1.0;
    long th = (long)floor(factor * (double)ch), tw = (long)floor(factor * (double)cw);
    if (budget > lo && th * tw < lo && th * tw > 0) {
        const double up = sqrt((double)lo / (double)(th * tw));
        th = (long)ceil(up * (double)th);
        tw = (long)ceil(up * (double)tw);
    }
    const long d = (long)s->merge;
    if (th % d) th = (th + (d - th % d)) * tw <= budget ? th + (d - th % d) : (th - th % d > d ? th - th % d : d);
    if (tw % d) tw = th * (tw + (d - tw % d)) <= budget ? tw + (d - tw % d) : (tw - tw % d > d ? tw - tw % d : d);
    *wp = th > 0 && tw > 0 ? (uint32_t)tw : 0u;
    *hp = th > 0 && tw > 0 ? (uint32_t)th : 0u;
}

static uint16_t f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

bool pulsar_nemotron_h_image_prepare(const pulsar_nemotron_h_vision_shape *s, const uint8_t *bytes, size_t len,
                                     pulsar_nemotron_h_image *img, char *err, size_t errlen) {
    uint8_t *rgb = NULL;
    int w = 0, h = 0;
    if (!vision_decode_rgb(bytes, len, &rgb, &w, &h)) {
        snprintf(err, errlen, "the image could not be decoded (PNG or JPEG)");
        return false;
    }
    pulsar_nemotron_h_tile(s, w, h, &img->wp, &img->hp);
    /* the processor divides by the tile's product when one side floors to zero (a page thousands of patches long
     * and one high): it raises; refused here by name */
    if (!img->wp || !img->hp || img->wp * img->hp > s->max_patches) {
        free(rgb);
        snprintf(err, errlen, "a %dx%d image has no tile within %u patches (its aspect ratio is past the processor's)",
                 w, h, s->max_patches);
        return false;
    }
    const int P = (int)s->patch, dw = (int)img->wp * P, dh = (int)img->hp * P;
    float *px = vision_torch_bicubic_aa_rgb(rgb, w, h, dw, dh);
    free(rgb);
    if (!px) {
        snprintf(err, errlen, "resizing %dx%d to %dx%d failed", w, h, dw, dh);
        return false;
    }
    const size_t plane = (size_t)dw * dh;
    img->pixels.assign(px, px + 3 * plane);
    free(px);
    for (int c = 0; c < 3; c++) {
        const float mean = s->mean[c], std = s->std[c];
        for (size_t i = 0; i < plane; i++) {
            float &v = img->pixels[(size_t)c * plane + i];
            v = (v / 255.0f - mean) / std;
        }
    }
    const uint32_t K = 3u * P * P;
    img->patches.assign((size_t)img->wp * img->hp * K, 0);
    for (uint32_t r = 0; r < img->hp; r++)
        for (uint32_t col = 0; col < img->wp; col++) {
            uint16_t *dst = img->patches.data() + (size_t)pulsar_nemotron_h_patch_slot(r, col, img->wp) * K;
            for (int c = 0; c < 3; c++)
                for (int y = 0; y < P; y++)
                    for (int x = 0; x < P; x++)
                        dst[(c * P + y) * P + x] =
                            f32_to_bf16(img->pixels[(size_t)c * plane + (size_t)(r * P + y) * dw + (col * P + x)]);
        }
    return true;
}

/* torch's compute_source_index_and_lambda (align_corners False, linear) for one axis of in -> out */
static void lin_taps(int i, int in, int out, int *i0, int *i1, float *l0, float *l1) {
    if (in == out) {
        *i0 = *i1 = i;
        *l0 = 1.0f;
        *l1 = 0.0f;
        return;
    }
    const float scale = (float)in / (float)out;
    float src = scale * ((float)i + 0.5f) - 0.5f;
    if (src < 0.0f) src = 0.0f;
    int k = (int)floorf(src);
    if (k > in - 1) k = in - 1;
    float l = src - (float)k;
    l = l < 0.0f ? 0.0f : l > 1.0f ? 1.0f : l;
    *i0 = k;
    *i1 = k + (k < in - 1 ? 1 : 0);
    *l1 = l;
    *l0 = 1.0f - l;
}

void pulsar_nemotron_h_positions(const pulsar_nemotron_h_vision_shape *s, const std::vector<float> &field, uint32_t wp,
                                 uint32_t hp, std::vector<float> *out) {
    const int G = (int)s->pos_grid, C = (int)s->width, M = (int)(wp > hp ? wp : hp);
    out->assign((size_t)wp * hp * C, 0.0f);
    for (uint32_t r = 0; r < hp; r++) {
        int y0, y1;
        float hy0, hy1;
        lin_taps((int)r, G, M, &y0, &y1, &hy0, &hy1);
        for (uint32_t c = 0; c < wp; c++) {
            int x0, x1;
            float wx0, wx1;
            lin_taps((int)c, G, M, &x0, &x1, &wx0, &wx1);
            const float w00 = hy0 * wx0, w01 = hy0 * wx1, w10 = hy1 * wx0, w11 = hy1 * wx1;
            const float *i00 = &field[((size_t)y0 * G + x0) * C], *i01 = &field[((size_t)y0 * G + x1) * C],
                        *i10 = &field[((size_t)y1 * G + x0) * C], *i11 = &field[((size_t)y1 * G + x1) * C];
            float *dst = out->data() + (size_t)pulsar_nemotron_h_patch_slot(r, c, wp) * C;
            for (int k = 0; k < C; k++) dst[k] = w00 * i00[k] + w01 * i01[k] + w10 * i10[k] + w11 * i11[k];
        }
    }
}

/* nemotron_parse_image.cpp -- Parse's page front (nemotron_parse.h section 3): the processor's pixels and the ViT's
 * position rows, on the host.
 *
 * THE PIXELS (hf_nemotron_parse_processor.py, NemotronParseImageProcessor.preprocess with navit_packing):
 *   decode     PNG / JPEG through the engine's codec (libpng / libjpeg-turbo: Pillow's own decoders)
 *   resize     _resize_with_aspect_ratio: aspect = w / h; past the canvas height h' = 2048, w' = int(h' * aspect);
 *              then past its width w' = 1664, h' = int(w' / aspect); Pillow BILINEAR (vision_pil_resize_rgb) when the
 *              size changed -- the content
 *   pad        _pad_to_navit_grid: white (255) at the bottom and right to multiples of 16 rows and 64 columns
 *   normalise  ToTensor's f32 p / 255, then (x - mean) / std in f32, rounded once to bf16
 *   patches    unfold 16 x 16, permuted (row, column, channel, y, x): element ch * 256 + y * 16 + x of patch
 *              (r, c) is pixel (16 r + y, 16 c + x) of channel ch
 * THE POSITIONS (hf_prefix_geometry.py _normalized_2d_position_pages, content-normalised): the 128 x 128 field
 *   resampled to the content's patch grid (ceil(ch / 16) x ceil(cw / 16)) -- torch's antialiased bilinear interpolate
 *   (align_corners False: Pillow's triangle filter with support max(1, scale), here in double), or adaptive average
 *   pooling when either side is one patch -- then grid_sample'd (bilinear, border, align_corners False; f32, as torch
 *   computes it) at each canvas patch's centre clamped to the content, normalised by the content size. */
#include "nemotron_parse.h"

#include <math.h>
#include <stdlib.h>

namespace {

/* torch's _compute_weights_aa for one output index: the taps [xmin, xmin + n) and their normalised weights */
void aa_taps(int i, int in, int out, int *xmin, int *n, double *w) {
    const double scale = (double)in / (double)out;
    const double support = scale >= 1.0 ? scale : 1.0, invscale = scale >= 1.0 ? 1.0 / scale : 1.0;
    const double center = scale * (i + 0.5);
    int lo = (int)(center - support + 0.5);
    if (lo < 0) lo = 0;
    int hi = (int)(center + support + 0.5);
    if (hi > in) hi = in;
    double tot = 0.0;
    for (int j = 0; j < hi - lo; j++) {
        const double x = fabs((j + lo - center + 0.5) * invscale);
        w[j] = x < 1.0 ? 1.0 - x : 0.0;
        tot += w[j];
    }
    for (int j = 0; j < hi - lo; j++) w[j] = tot != 0.0 ? w[j] / tot : 0.0;
    *xmin = lo;
    *n = hi - lo;
}

/* the field [G][G][C] (row, column, channel) -> [r][c][C] at the content grid */
void resample_field(const std::vector<float> &f, int G, int C, int r, int c, std::vector<float> *out) {
    out->assign((size_t)r * c * C, 0.0f);
    if (r == 1 || c == 1) {   /* adaptive_avg_pool2d: start floor(i G / n), end ceil((i + 1) G / n) */
        for (int oy = 0; oy < r; oy++)
            for (int ox = 0; ox < c; ox++) {
                const int y0 = oy * G / r, y1 = ((oy + 1) * G + r - 1) / r;
                const int x0 = ox * G / c, x1 = ((ox + 1) * G + c - 1) / c;
                const double cnt = (double)(y1 - y0) * (x1 - x0);
                for (int ch = 0; ch < C; ch++) {
                    double s = 0.0;
                    for (int y = y0; y < y1; y++)
                        for (int x = x0; x < x1; x++) s += f[((size_t)y * G + x) * C + ch];
                    (*out)[((size_t)oy * c + ox) * C + ch] = (float)(s / cnt);
                }
            }
        return;
    }
    const int maxt = 2 * (G / (r < c ? r : c) + 2) + 2;
    std::vector<double> wy((size_t)maxt), wx((size_t)c * maxt);
    std::vector<int> x0(c), nx(c);
    for (int ox = 0; ox < c; ox++) aa_taps(ox, G, c, &x0[ox], &nx[ox], &wx[(size_t)ox * maxt]);
    std::vector<double> acc(C);
    for (int oy = 0; oy < r; oy++) {
        int y0, ny;
        aa_taps(oy, G, r, &y0, &ny, wy.data());
        for (int ox = 0; ox < c; ox++) {
            std::fill(acc.begin(), acc.end(), 0.0);
            for (int a = 0; a < ny; a++)
                for (int bb = 0; bb < nx[ox]; bb++) {
                    const double wt = wy[a] * wx[(size_t)ox * maxt + bb];
                    if (wt == 0.0) continue;
                    const float *src = &f[((size_t)(y0 + a) * G + (x0[ox] + bb)) * C];
                    for (int ch = 0; ch < C; ch++) acc[ch] += wt * src[ch];
                }
            float *dst = &(*out)[((size_t)oy * c + ox) * C];
            for (int ch = 0; ch < C; ch++) dst[ch] = (float)acc[ch];
        }
    }
}

/* one axis of the sampling grid: patch i's centre clamped to the content, normalised, as grid_sample's source index */
float grid_index(uint32_t i, uint32_t patch, uint32_t content, uint32_t n_content) {
    const float start = (float)(i * patch);
    float extent = (float)content - start;
    extent = extent < 0.0f ? 0.0f : extent > (float)patch ? (float)patch : extent;
    float centre = start + 0.5f * extent;
    if (centre > (float)content) centre = (float)content;
    const float norm = centre / (float)content;
    const float g = 2.0f * norm - 1.0f;
    float ix = ((g + 1.0f) * (float)n_content - 1.0f) / 2.0f;   /* unnormalise, align_corners False */
    return fminf((float)(n_content - 1), fmaxf(ix, 0.0f));      /* border */
}

}  // namespace

bool pulsar_nemotron_parse_page_prepare(const pulsar_nemotron_parse_shape *s, const uint8_t *bytes, size_t len,
                                        pulsar_nemotron_parse_page *page, char *err, size_t errlen) {
    uint8_t *rgb = NULL;
    int w = 0, h = 0;
    if (!vision_decode_rgb(bytes, len, &rgb, &w, &h) || w <= 0 || h <= 0) {
        if (err) snprintf(err, errlen, "the image is not a PNG or JPEG this engine decodes");
        return false;
    }
    /* _resize_with_aspect_ratio (LongestMaxSizeHW): the original's own int truncations */
    const double aspect = (double)w / (double)h;
    int nh = h, nw = w;
    if (nh > (int)s->canvas_h) {
        nh = (int)s->canvas_h;
        nw = (int)(nh * aspect);
    }
    if (nw > (int)s->canvas_w) {
        nw = (int)s->canvas_w;
        nh = (int)(nw / aspect);
    }
    if (nw <= 0 || nh <= 0) {
        free(rgb);
        if (err) snprintf(err, errlen, "a %d x %d image resizes to nothing", w, h);
        return false;
    }
    if (nw != w || nh != h) {
        uint8_t *r = vision_pil_resize_rgb(rgb, w, h, nw, nh, PULSAR_PIL_BILINEAR);
        free(rgb);
        if (!r) {
            if (err) snprintf(err, errlen, "the resize failed");
            return false;
        }
        rgb = r;
    }
    const uint32_t P = s->patch;
    page->ch = (uint32_t)nh;
    page->cw = (uint32_t)nw;
    page->h = ((uint32_t)nh + s->align_h - 1) / s->align_h * s->align_h;
    page->w = ((uint32_t)nw + s->align_w - 1) / s->align_w * s->align_w;
    const uint32_t rows = page->h / P, cols = page->w / P, E = 3u * P * P;
    page->patches.assign((size_t)rows * cols * E, 0);
    float lut[3][256];
    for (int ch = 0; ch < 3; ch++)
        for (int p = 0; p < 256; p++) {
            const float x = (float)p / 255.0f;
            lut[ch][p] = (x - s->mean[ch]) / s->std[ch];
        }
    auto bf16 = [](float v) {
        uint32_t u;
        memcpy(&u, &v, 4);
        u += 0x7fffu + ((u >> 16) & 1u);
        return (uint16_t)(u >> 16);
    };
    for (uint32_t y = 0; y < page->h; y++)
        for (uint32_t x = 0; x < page->w; x++) {
            const bool in = y < (uint32_t)nh && x < (uint32_t)nw;
            const uint8_t *px = in ? rgb + ((size_t)y * nw + x) * 3u : NULL;
            uint16_t *patch = &page->patches[((size_t)(y / P) * cols + x / P) * E];
            for (int ch = 0; ch < 3; ch++)
                patch[(size_t)ch * P * P + (y % P) * P + x % P] = bf16(lut[ch][in ? px[ch] : 255]);
        }
    free(rgb);
    return true;
}

void pulsar_nemotron_parse_positions(const pulsar_nemotron_parse_shape *s, const std::vector<float> &field, uint32_t h,
                                     uint32_t w, uint32_t ch, uint32_t cw, std::vector<float> *out) {
    const uint32_t P = s->patch, G = s->pos_grid, C = s->vit_width;
    const uint32_t rows = h / P, cols = w / P, cr = (ch + P - 1) / P, cc = (cw + P - 1) / P;
    std::vector<float> grid;
    resample_field(field, (int)G, (int)C, (int)cr, (int)cc, &grid);
    out->assign((size_t)rows * cols * C, 0.0f);
    for (uint32_t r = 0; r < rows; r++) {
        const float iy = grid_index(r, P, ch, cr);
        const int y0 = (int)floorf(iy);
        const float ty = iy - (float)y0;
        for (uint32_t c = 0; c < cols; c++) {
            const float ix = grid_index(c, P, cw, cc);
            const int x0 = (int)floorf(ix);
            const float tx = ix - (float)x0;
            const float wts[4] = {(1.0f - tx) * (1.0f - ty), tx * (1.0f - ty), (1.0f - tx) * ty, tx * ty};
            const int ys[4] = {y0, y0, y0 + 1, y0 + 1}, xs[4] = {x0, x0 + 1, x0, x0 + 1};
            float *dst = &(*out)[((size_t)r * cols + c) * C];
            for (int k = 0; k < 4; k++) {
                if (ys[k] < 0 || ys[k] >= (int)cr || xs[k] < 0 || xs[k] >= (int)cc || wts[k] == 0.0f) continue;
                const float *src = &grid[((size_t)ys[k] * cc + xs[k]) * C];
                for (uint32_t d = 0; d < C; d++) dst[d] += wts[k] * src[d];
            }
        }
    }
}

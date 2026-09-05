/*
 * DS4 quantization facade.
 *
 * These are the small GGUF quantization pieces needed by our DeepSeek V4
 * Flash recipes: float conversion, q2_K, iq2_xxs, fp8_e4m3, and mxfp4.  The code is
 * local C and deliberately narrow; other GGUF type IDs are named for metadata
 * compatibility, but cannot be emitted by this tool.
 *
 * The quantized block layouts and search procedures are derived from the
 * MIT-licensed GGML/llama.cpp quantizers.  Keep changes conservative: byte
 * layout compatibility is more important here than generality.
 */

#include "quants_internal.h"

pthread_mutex_t ds4q_init_mutex = PTHREAD_MUTEX_INITIALIZER;

const ds4q_traits ds4q_type_traits[DS4Q_TYPE_COUNT] = {
    [DS4Q_TYPE_F32]     = { "f32",      1,   4, false, false },
    [DS4Q_TYPE_F16]     = { "f16",      1,   2, false, false },
    [DS4Q_TYPE_Q4_0]    = { "q4_0",    32,  18, false, false },
    [DS4Q_TYPE_Q4_1]    = { "q4_1",    32,  20, false, false },
    [DS4Q_TYPE_Q5_0]    = { "q5_0",    32,  22, false, false },
    [DS4Q_TYPE_Q5_1]    = { "q5_1",    32,  24, false, false },
    [DS4Q_TYPE_Q8_0]    = { "q8_0",    32,  34, false, false },
    [DS4Q_TYPE_Q8_1]    = { "q8_1",    32,  36, false, false },
    [DS4Q_TYPE_Q2_K]    = { "q2_K",  QK_K,  84, true,  false },
    [DS4Q_TYPE_Q3_K]    = { "q3_K",  QK_K, 110, false, false },
    [DS4Q_TYPE_Q5_K]    = { "q5_K",  QK_K, 176, false, false },
    [DS4Q_TYPE_Q6_K]    = { "q6_K",  QK_K, 210, false, false },
    [DS4Q_TYPE_Q8_K]    = { "q8_K",  QK_K, 292, false, false },
    [DS4Q_TYPE_FP8_E4M3]= { "fp8_e4m3", 32,  33, true,  false },
    [DS4Q_TYPE_IQ2_XXS] = { "iq2_xxs", QK_K,  66, true,  true  },
    /* Same block size and type size as type 16 -- SoA is a pure permutation of
     * the same 66 B/block, so ds4q_row_size() and every byte accounting path
     * work unchanged.  Not a quantizer target itself: produced by repacking a
     * type-16 tensor, so requires_imatrix is inherited at the source step. */
    [DS4Q_TYPE_IQ2_XXS_SOA] = { "iq2_xxs_soa", QK_K, 66, true, false },
    [DS4Q_TYPE_IQ2_XS]  = { "iq2_xs",  QK_K,  74, false, true  },
    [DS4Q_TYPE_IQ3_XXS] = { "iq3_xxs", QK_K,  98, false, false },
    [DS4Q_TYPE_IQ1_S]   = { "iq1_s",   QK_K,  50, false, true  },
    [DS4Q_TYPE_IQ4_NL]  = { "iq4_nl",     32,  18, false, false },
    [DS4Q_TYPE_IQ3_S]   = { "iq3_s",   QK_K, 110, false, false },
    [DS4Q_TYPE_IQ2_S]   = { "iq2_s",   QK_K,  82, false, false },
    [DS4Q_TYPE_IQ4_XS]  = { "iq4_xs",  QK_K, 136, false, false },
    [DS4Q_TYPE_I8]      = { "i8",          1,   1, false, false },
    [DS4Q_TYPE_I16]     = { "i16",         1,   2, false, false },
    [DS4Q_TYPE_I32]     = { "i32",         1,   4, false, false },
    [DS4Q_TYPE_I64]     = { "i64",         1,   8, false, false },
    [DS4Q_TYPE_F64]     = { "f64",         1,   8, false, false },
    [DS4Q_TYPE_IQ1_M]   = { "iq1_m",   QK_K,  56, false, false },
    [DS4Q_TYPE_BF16]    = { "bf16",        1,   2, false, false },
    [DS4Q_TYPE_TQ1_0]   = { "tq1_0",   QK_K,  54, false, false },
    [DS4Q_TYPE_TQ2_0]   = { "tq2_0",   QK_K,  66, false, false },
    [DS4Q_TYPE_MXFP4]   = { "mxfp4",      32,  17, true,  false },
    /* type_size 0: not a fixed-stride row format (per-expert data + swizzled
     * SF tile); size via ds4q_cutlass_mxfp4_bytes().  ds4q_row_size returns 0
     * and ds4q_can_quantize false by design -- the expert generation path
     * packs it explicitly via ds4q_pack_cutlass_mxfp4. */
    [DS4Q_TYPE_CUTLASS_MXFP4] = { "cutlass_mxfp4", 32, 0, false, false },
    /* MXFP8_LT shares the type-38 {32,33} geometry: for the 128-aligned
     * workhorse shapes it targets, the de-interleaved E4M3 data plus the
     * padded swizzled SF tile total exactly KB*33 bytes/row, so ds4q_row_size
     * gives the right slot size.  can_quantize is false -- f32_to_type packs it
     * explicitly via ds4q_pack_mxfp8_lt after a type-38 quantize pass. */
    [DS4Q_TYPE_MXFP8_LT] = { "mxfp8_lt", 32, 33, false, false },
    /* IQ2_XXS_MMQ shares the type-16 {QK_K,66} geometry exactly -- it is the
     * same bytes permuted into two planes, and the permutation is size-
     * preserving whenever nblk%32==0 (true for every shipped expert shape), so
     * ds4q_row_size gives the right slot size.  can_quantize is false: the
     * expert path quantizes to raw IQ2_XXS first, then packs explicitly via
     * ds4q_pack_iq2_xxs_mmq -- the same two-step f32_to_type does for
     * MXFP8_LT.  Distinct permutation from IQ2_XXS_SOA (42); not
     * interchangeable. */
    [DS4Q_TYPE_IQ2_XXS_MMQ] = { "iq2_xxs_mmq", QK_K, 66, false, false },
};

float ds4q_f32_from_bits(uint32_t bits) {
    union {
        uint32_t u;
        float f;
    } v = { .u = bits };
    return v.f;
}

uint32_t ds4q_f32_to_bits(float f) {
    union {
        float f;
        uint32_t u;
    } v = { .f = f };
    return v.u;
}

uint16_t ds4q_f32_to_f16(float f) {
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = ds4q_f32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) bias = UINT32_C(0x71000000);

    base = ds4q_f32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t out = ds4q_f32_to_bits(base);
    const uint32_t exp_bits = (out >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = out & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (uint16_t)((sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign));
}

int ds4q_nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i;
    memcpy(&i, &val, sizeof(i));
    return (i & 0x007fffff) - 0x00400000;
}

float ds4q_make_qkx2_quants(int n, int nmax, const float *x, const float *weights,
                                   uint8_t *L, float *the_min, uint8_t *Laux,
                                   float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) {
        memset(L, 0, (size_t)n);
        *the_min = -min;
        return 0.0f;
    }
    float iscale = nmax / (max - min);
    float scale = 1 / iscale;
    float best_error = 0;
    for (int i = 0; i < n; i++) {
        int l = ds4q_nearest_int(iscale * (x[i] - min));
        L[i] = DS4Q_MAX(0, DS4Q_MIN(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        best_error += weights[i] * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; is++) {
        if (max <= min) break;
        iscale = (rmin + rdelta * is + nmax) / (max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; i++) {
            int l = ds4q_nearest_int(iscale * (x[i] - min));
            l = DS4Q_MAX(0, DS4Q_MIN(nmax, l));
            Laux[i] = l;
            float w = weights[i];
            sum_l += w * l;
            sum_l2 += w * l * l;
            sum_xl += w * l * x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
            float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float cur_error = 0;
            for (int i = 0; i < n; i++) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                cur_error += weights[i] * diff;
            }
            if (cur_error < best_error) {
                memcpy(L, Laux, (size_t)n);
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

float ds4q_make_qkx3_quants(int n, int nmax, const float *x, const float *weights,
                                   uint8_t *L, float *the_min, uint8_t *Laux,
                                   float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights ? weights[0] : x[0] * x[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights ? weights[i] : x[i] * x[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max <= min) {
        memset(L, 0, (size_t)n);
        *the_min = -min;
        return 0.0f;
    }
    float iscale = nmax / (max - min);
    float scale = 1 / iscale;
    float best_mad = 0;
    for (int i = 0; i < n; i++) {
        int l = ds4q_nearest_int(iscale * (x[i] - min));
        L[i] = DS4Q_MAX(0, DS4Q_MIN(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights ? weights[i] : x[i] * x[i];
        best_mad += w * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; is++) {
        if (max <= min) break;
        iscale = (rmin + rdelta * is + nmax) / (max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; i++) {
            int l = ds4q_nearest_int(iscale * (x[i] - min));
            l = DS4Q_MAX(0, DS4Q_MIN(nmax, l));
            Laux[i] = l;
            float w = weights ? weights[i] : x[i] * x[i];
            sum_l += w * l;
            sum_l2 += w * l * l;
            sum_xl += w * l * x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
            float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float mad = 0;
            for (int i = 0; i < n; i++) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                float w = weights ? weights[i] : x[i] * x[i];
                mad += w * diff;
            }
            if (mad < best_mad) {
                memcpy(L, Laux, (size_t)n);
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

float ds4q_make_qp_quants(int n, int nmax, const float *x, uint8_t *L, const float *quant_weights) {
    float max = 0;
    for (int i = 0; i < n; i++) max = DS4Q_MAX(max, x[i]);
    if (max < DS4Q_GROUP_MAX_EPS) {
        memset(L, 0, (size_t)n);
        return 0.0f;
    }
    float iscale = nmax / max;
    for (int i = 0; i < n; i++) L[i] = ds4q_nearest_int(iscale * x[i]);
    float scale = 1 / iscale;
    float best_mse = 0;
    for (int i = 0; i < n; i++) {
        float diff = x[i] - scale * L[i];
        best_mse += quant_weights[i] * diff * diff;
    }
    for (int is = -4; is <= 4; is++) {
        if (is == 0) continue;
        float iscale_is = (0.1f * is + nmax) / max;
        float scale_is = 1 / iscale_is;
        float mse = 0;
        for (int i = 0; i < n; i++) {
            int l = ds4q_nearest_int(iscale_is * x[i]);
            l = DS4Q_MIN(nmax, l);
            float diff = x[i] - scale_is * l;
            mse += quant_weights[i] * diff * diff;
        }
        if (mse < best_mse) {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    float sumlx = 0, suml2 = 0;
    for (int i = 0; i < n; i++) {
        int l = ds4q_nearest_int(iscale * x[i]);
        l = DS4Q_MIN(nmax, l);
        L[i] = l;
        float w = quant_weights[i];
        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }
    for (int itry = 0; itry < 5; itry++) {
        int n_changed = 0;
        for (int i = 0; i < n; i++) {
            float w = quant_weights[i];
            float slx = sumlx - w * x[i] * L[i];
            float sl2 = suml2 - w * L[i] * L[i];
            if (slx > 0 && sl2 > 0) {
                int new_l = ds4q_nearest_int(x[i] * sl2 / slx);
                new_l = DS4Q_MIN(nmax, new_l);
                if (new_l != L[i]) {
                    slx += w * x[i] * new_l;
                    sl2 += w * new_l * new_l;
                    if (slx * slx * suml2 > sumlx * sumlx * sl2) {
                        L[i] = new_l;
                        sumlx = slx;
                        suml2 = sl2;
                        n_changed++;
                    }
                }
            }
        }
        if (!n_changed) break;
    }
    return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
}

const char *ds4q_type_name(ds4q_type type) {
    if (type < 0 || type >= DS4Q_TYPE_COUNT) return NULL;
    return ds4q_type_traits[type].name;
}

bool ds4q_can_quantize(ds4q_type type) {
    if (type < 0 || type >= DS4Q_TYPE_COUNT) return false;
    return ds4q_type_traits[type].can_quantize;
}

int64_t ds4q_block_size(ds4q_type type) {
    if (type < 0 || type >= DS4Q_TYPE_COUNT) return 0;
    return ds4q_type_traits[type].block_size;
}

size_t ds4q_row_size(ds4q_type type, int64_t ne) {
    if (type < 0 || type >= DS4Q_TYPE_COUNT) return 0;
    const ds4q_traits *tr = &ds4q_type_traits[type];
    if (tr->block_size <= 0 || tr->type_size == 0 || ne % tr->block_size != 0) return 0;
    return tr->type_size * (size_t)(ne / tr->block_size);
}

bool ds4q_requires_imatrix(ds4q_type type) {
    if (type < 0 || type >= DS4Q_TYPE_COUNT) return false;
    return ds4q_type_traits[type].requires_imatrix;
}


void ds4q_quantize_init(ds4q_type type) {
    if (type == DS4Q_TYPE_IQ2_XXS) {
        ds4q_iq2_xxs_init();
    }
}

size_t ds4q_quantize_chunk(ds4q_type type, const float *src, void *dst,
                           int64_t start, int64_t nrows, int64_t ncols,
                           const float *imatrix) {
    if (type == DS4Q_TYPE_Q2_K) {
        return ds4q_quantize_q2_k(src, dst, start, nrows, ncols, imatrix);
    }
    if (type == DS4Q_TYPE_IQ2_XXS) {
        return ds4q_quantize_iq2_xxs(src, dst, start, nrows, ncols, imatrix);
    }
    if (type == DS4Q_TYPE_FP8_E4M3) {
        (void)imatrix;
        return ds4q_quantize_fp8_e4m3(src, dst, start, nrows, ncols);
    }
    if (type == DS4Q_TYPE_MXFP4) {
        (void)imatrix;
        return ds4q_quantize_mxfp4(src, dst, start, nrows, ncols);
    }
    (void)src;
    (void)dst;
    (void)start;
    (void)nrows;
    (void)ncols;
    (void)imatrix;
    assert(!"unsupported DS4 quantization target");
    return 0;
}

float ds4q_f16_to_f32(uint16_t bits) {
    const uint32_t w = (uint32_t)bits << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;
    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
    const float exp_scale = 0x1.0p-112f;
    const float normalized_value = ds4q_f32_from_bits((two_w >> 4) + exp_offset) * exp_scale;
    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = ds4q_f32_from_bits((two_w >> 17) | magic_mask) - magic_bias;
    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? ds4q_f32_to_bits(denormalized_value) : ds4q_f32_to_bits(normalized_value));
    return ds4q_f32_from_bits(result);
}

float ds4q_bf16_to_f32(uint16_t bits) {
    return ds4q_f32_from_bits((uint32_t)bits << 16);
}

void ds4q_f32_to_f16_row(const float *src, uint16_t *dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) dst[i] = ds4q_f32_to_f16(src[i]);
}

void ds4q_f32_to_bf16_row(const float *src, uint16_t *dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        uint32_t bits = ds4q_f32_to_bits(src[i]);
        if ((bits & UINT32_C(0x7fffffff)) > UINT32_C(0x7f800000)) {
            dst[i] = (uint16_t)((bits >> 16) | 64);
        } else {
            dst[i] = (uint16_t)((bits + (UINT32_C(0x7fff) + ((bits >> 16) & 1))) >> 16);
        }
    }
}


/* ---- IQ2_XXS_SOA (type 42) repack ------------------------------------------
 * Byte spec lives in quants.h.  q plane first (block b at b*64), then the d
 * plane (block b at nblk*64 + b*2).  Total nblk*66 == the packed size.
 *
 * The two directions are exact inverses and MUST stay so: the engine loads the
 * SoA artifact directly, and any host-side dequant/diff path round-trips
 * through unpack.  block_iq2_xxs is { uint16 d; uint16 qs[32]; } == 66 B, and
 * that 2-byte lead is precisely what forces the misaligned code stream. */
size_t ds4q_iq2_xxs_soa_qplane_bytes(int64_t ne) {
    if (ne <= 0 || ne % QK_K != 0) return 0;
    return (size_t)(ne / QK_K) * 64u;
}

void ds4q_iq2_xxs_soa_repack(const void *packed, void *soa, int64_t ne) {
    if (ne <= 0 || ne % QK_K != 0) return;
    const int64_t nblk = ne / QK_K;
    const uint8_t *src = (const uint8_t *)packed;
    uint8_t *q = (uint8_t *)soa;                     /* [0, nblk*64) */
    uint8_t *d = (uint8_t *)soa + (size_t)nblk * 64u; /* [nblk*64, +nblk*2) */
    for (int64_t b = 0; b < nblk; b++) {
        const uint8_t *sb = src + (size_t)b * 66u;
        memcpy(d + (size_t)b * 2u, sb, 2u);          /* uint16 d */
        memcpy(q + (size_t)b * 64u, sb + 2u, 64u);   /* uint16 qs[32] */
    }
}

void ds4q_iq2_xxs_soa_unpack(const void *soa, void *packed, int64_t ne) {
    if (ne <= 0 || ne % QK_K != 0) return;
    const int64_t nblk = ne / QK_K;
    const uint8_t *q = (const uint8_t *)soa;
    const uint8_t *d = (const uint8_t *)soa + (size_t)nblk * 64u;
    uint8_t *dst = (uint8_t *)packed;
    for (int64_t b = 0; b < nblk; b++) {
        uint8_t *db = dst + (size_t)b * 66u;
        memcpy(db, d + (size_t)b * 2u, 2u);
        memcpy(db + 2u, q + (size_t)b * 64u, 64u);
    }
}

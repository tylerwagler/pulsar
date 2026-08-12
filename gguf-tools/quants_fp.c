#include "quants_internal.h"

#include <stdio.h>
#include <stdlib.h>

/* A NaN weight means the source tensor is corrupt; encoding it (E4M3 NaN or
 * a garbage E2M1 pick) would poison inference far from the cause. Die loudly
 * at quantization time instead. */
static void ds4q_reject_nan(const float *x, int n, const char *what) {
    for (int j = 0; j < n; j++) {
        if (x[j] != x[j]) {
            fprintf(stderr, "ds4-quantize: NaN weight in %s input block; source tensor is corrupt\n", what);
            exit(1);
        }
    }
}

/* ---- MXFP8: E4M3 values + per-32 E8M0 block scale; block = 33 B / 32 elems
 * (8.25 bpw). byte[0] = E8M0 scale (value = 2^(b-127)); byte[1..32] = E4M3.
 * Hardware-native on Blackwell MX tensor cores (cuBLASLt VEC32_UE8M0). For the
 * FP8-source attn/dense/shared weights this is near-lossless and >= Q8_0. ---- */
static uint8_t ds4q_f32_to_e4m3(float x) {
    if (x != x) return 0x7f;                          /* NaN */
    uint8_t s = (x < 0.0f) ? 0x80 : 0x00;
    float a = fabsf(x);
    if (a >= 448.0f) return (uint8_t)(s | 0x7e);      /* saturate (incl inf) */
    if (a == 0.0f)   return s;                        /* signed zero */
    union { float f; uint32_t u; } v; v.f = a;
    int32_t exp = (int32_t)((v.u >> 23) & 0xff) - 127;
    uint32_t sig = (1u << 23) | (v.u & 0x7fffffu);    /* 24-bit significand */
    int32_t e8 = exp + 7;                             /* E4M3 biased exponent */
    if (e8 <= 0) {                                    /* subnormal: code * 2^-9 */
        int code = (int)lrintf(a * 512.0f);
        if (code > 7) return (uint8_t)(s | (1u << 3));   /* round up to min normal */
        return (uint8_t)(s | (code & 0x7));
    }
    uint32_t mant3 = (sig >> 20) & 0x7;
    uint32_t rem = sig & ((1u << 20) - 1), half = 1u << 19;
    if (rem > half || (rem == half && (mant3 & 1))) {     /* round to nearest even */
        if (++mant3 == 8) { mant3 = 0; e8++; }
    }
    if (e8 > 15 || (e8 == 15 && mant3 >= 7)) return (uint8_t)(s | 0x7e); /* saturate */
    return (uint8_t)(s | ((e8 & 0xf) << 3) | (mant3 & 0x7));
}

static float ds4q_e4m3_to_f32(uint8_t b) {
    float sign = (b & 0x80) ? -1.0f : 1.0f;
    uint32_t e = (b >> 3) & 0xf, m = b & 0x7;
    if (e == 0) return sign * (float)m * (1.0f / 512.0f);          /* subnormal */
    if (e == 15 && m == 7) return sign * (float)NAN;
    return sign * (1.0f + (float)m * 0.125f) * ldexpf(1.0f, (int)e - 7);
}

size_t ds4q_quantize_fp8_e4m3(const float *src, void *dst, int64_t start,
                                     int64_t nrows, int64_t ncols) {
    const int64_t qk = 32;
    const size_t row_size = ds4q_row_size(DS4Q_TYPE_FP8_E4M3, ncols);
    const int64_t start_row = start / ncols;
    uint8_t *out = (uint8_t *)dst + (size_t)start_row * row_size;
    const int64_t nblocks = nrows * (ncols / qk);
    for (int64_t b = 0; b < nblocks; b++) {
        const float *x = src + start + (size_t)b * qk;
        ds4q_reject_nan(x, (int)qk, "MXFP8");
        float amax = 0.0f;
        for (int j = 0; j < qk; j++) { float av = fabsf(x[j]); if (av > amax) amax = av; }
        int scale_exp = -127;
        if (amax > 0.0f) { int e; frexpf(amax, &e); scale_exp = (e - 1) - 7; } /* max elem -> [128,256) */
        /* Clamp keeps the E8M0 byte <= 254: 255 would bitcast to +Inf in the
         * engine's ldexp/bit-shift scale decode. */
        if (scale_exp < -127) scale_exp = -127;
        if (scale_exp >  127) scale_exp =  127;
        out[0] = (uint8_t)(scale_exp + 127);            /* E8M0 */
        const float inv = ldexpf(1.0f, -scale_exp);
        for (int j = 0; j < qk; j++) out[1 + j] = ds4q_f32_to_e4m3(x[j] * inv);
        out += 33;
    }
    return (size_t)nrows * row_size;
}

void ds4q_dequantize_fp8_e4m3(const void *blocks, float *out, int64_t n) {
    const uint8_t *b = (const uint8_t *)blocks;
    const int64_t nblocks = n / 32;
    for (int64_t i = 0; i < nblocks; i++) {
        const float scale = ldexpf(1.0f, (int)b[0] - 127);
        for (int j = 0; j < 32; j++) out[i * 32 + j] = ds4q_e4m3_to_f32(b[1 + j]) * scale;
        b += 33;
    }
}

/* ---- MXFP4: E2M1 values + per-32 E8M0 block scale; block = 17 B / 32 elems
 * (4.25 bpw). byte[0] = E8M0 scale (value = 2^(b-127)); byte[1..16] = packed
 * nibbles: byte[1+j] = v[2j] | (v[2j+1] << 4).  E2M1 representable magnitudes
 * are {0, 0.5, 1, 1.5, 2, 3, 4, 6}, sign via bit 3.  This format matches
 * DS4_TENSOR_FP4_E2M1 in the runtime. ---- */
static const float ds4q_e2m1_mag[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };

static uint8_t ds4q_f32_to_e2m1(float x) {
    uint8_t s = (x < 0.0f) ? 0x08 : 0x00; /* sign lives at bit 3 of the nibble */
    float a = fabsf(x);
    if (a == 0.0f) return s;
    if (a >= 6.0f) return (uint8_t)(s | 0x07);
    int best = 0;
    float best_diff = fabsf(a - ds4q_e2m1_mag[0]);
    for (int i = 1; i < 8; i++) {
        float d = fabsf(a - ds4q_e2m1_mag[i]);
        if (d < best_diff) { best = i; best_diff = d; }
    }
    return (uint8_t)(s | (best & 0x7));
}

size_t ds4q_quantize_mxfp4(const float *src, void *dst, int64_t start,
                                   int64_t nrows, int64_t ncols) {
    const int64_t qk = 32;
    const size_t row_size = ds4q_row_size(DS4Q_TYPE_MXFP4, ncols);
    const int64_t start_row = start / ncols;
    uint8_t *out = (uint8_t *)dst + (size_t)start_row * row_size;
    const int64_t nblocks = nrows * (ncols / qk);
    for (int64_t b = 0; b < nblocks; b++) {
        const float *x = src + start + (size_t)b * qk;
        ds4q_reject_nan(x, (int)qk, "MXFP4");
        float amax = 0.0f;
        for (int j = 0; j < qk; j++) { float av = fabsf(x[j]); if (av > amax) amax = av; }
        int scale_exp = -127;
        if (amax > 0.0f) scale_exp = (int)ceilf(log2f(amax / 6.0f));
        /* Same E8M0 <= 254 invariant as the MXFP8 writer above. */
        if (scale_exp < -127) scale_exp = -127;
        if (scale_exp >  127) scale_exp =  127;
        out[0] = (uint8_t)(scale_exp + 127);
        const float inv = ldexpf(1.0f, -scale_exp);
        for (int j = 0; j < 16; j++) {
            uint8_t lo = ds4q_f32_to_e2m1(x[2*j]     * inv);
            uint8_t hi = ds4q_f32_to_e2m1(x[2*j + 1] * inv);
            out[1 + j] = (lo & 0x0f) | (hi << 4);
        }
        out += 17;
    }
    return (size_t)nrows * row_size;
}

static inline float ds4q_e2m1_to_f32(uint8_t nib) {
    const float mag = ds4q_e2m1_mag[nib & 0x07];
    return (nib & 0x08) ? -mag : mag;
}

void ds4q_dequantize_mxfp4(const void *blocks, float *out, int64_t n) {
    const uint8_t *b = (const uint8_t *)blocks;
    const int64_t nblocks = n / 32;
    for (int64_t i = 0; i < nblocks; i++) {
        const float scale = ldexpf(1.0f, (int)b[0] - 127);
        for (int j = 0; j < 16; j++) {
            out[i * 32 + 2*j]     = ds4q_e2m1_to_f32(b[1 + j] & 0x0f) * scale;
            out[i * 32 + 2*j + 1] = ds4q_e2m1_to_f32(b[1 + j] >> 4)   * scale;
        }
        b += 17;
    }
}

/* ---- CUTLASS_MXFP4 (type 40): the grouped tensor-core MoE weight layout.
 * Per expert of shape [N=nrows(out), K=ncols(in)]:
 *   data: N*K/2 bytes -- the source E2M1 [N, K/2] nibble array verbatim
 *         (K-major sub-byte order matches CUTLASS ColumnMajor B packing);
 *   SF:   swizzled E8M0 scale tile, one byte per (row, 32-elem K-block),
 *         Blackwell 128x4 tile-atom layout (rows padded to 128, K-blocks
 *         padded to 4, padding zero-filled).
 * The swizzle is the same one the engine uses for cuBLASLt MX block scales
 * (mx_sfoff in src/cuda/ds4_cuda_matmul.cu) and matches CUTLASS
 * Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB. ---- */
size_t ds4q_mx_sfoff(int64_t row, int64_t kb, int64_t kbp) {
    return (size_t)((row / 128) * (kbp / 4) + kb / 4) * 512
         + (size_t)(row % 32) * 16
         + (size_t)((row % 128) / 32) * 4
         + (size_t)(kb % 4);
}

size_t ds4q_cutlass_mxfp4_sf_bytes(int64_t nrows, int64_t ncols) {
    const int64_t rows_pad = (nrows + 127) / 128 * 128;
    const int64_t kb_pad = (ncols / 32 + 3) / 4 * 4;
    return (size_t)rows_pad * (size_t)kb_pad;
}

size_t ds4q_cutlass_mxfp4_bytes(int64_t nrows, int64_t ncols) {
    return (size_t)nrows * (size_t)ncols / 2 + ds4q_cutlass_mxfp4_sf_bytes(nrows, ncols);
}

void ds4q_pack_cutlass_mxfp4(const uint8_t *e2m1, const uint8_t *e8m0,
                             void *dst, int64_t nrows, int64_t ncols) {
    if (ncols % 32 != 0) {
        fprintf(stderr, "ds4-quantize: cutlass_mxfp4 ncols %lld is not divisible by 32\n",
                (long long)ncols);
        exit(1);
    }
    const size_t data_bytes = (size_t)nrows * (size_t)ncols / 2;
    memcpy(dst, e2m1, data_bytes);
    uint8_t *sf = (uint8_t *)dst + data_bytes;
    const int64_t kb_n = ncols / 32;
    const int64_t kb_pad = (kb_n + 3) / 4 * 4;
    memset(sf, 0, ds4q_cutlass_mxfp4_sf_bytes(nrows, ncols));
    for (int64_t r = 0; r < nrows; r++) {
        for (int64_t kb = 0; kb < kb_n; kb++) {
            sf[ds4q_mx_sfoff(r, kb, kb_pad)] = e8m0[(size_t)r * (size_t)kb_n + (size_t)kb];
        }
    }
}

/* ---- MXFP8_LT (type 41): the pre-swizzled FP8 workhorse layout.  Mirrors the
 * device buffer the runtime otherwise builds at first use from a type-38 weight
 * (see DS4_TENSOR_MXFP8_LT).  Input is the type-38 payload for a tensor of shape
 * [nrows=out, ncols=in]: nrows*KB blocks of [E8M0][32 x E4M3] (KB = in/32).
 * Output is the E4M3 data de-interleaved to [out,in] row-major (== [in,out]
 * col-major) followed by the mx_sfoff-swizzled E8M0 scale, zero-padded to
 * rup(out,128) x KBp.  Byte-identical to repack_tensor() in
 * tools/mxfp8_prestore/repack_mxfp8_lt.py.  For 128-aligned shapes
 * (out%128==0, KB%4==0 -- every shipped workhorse) the total equals the type-38
 * size exactly, so it is an in-place layout change. ---- */
size_t ds4q_mxfp8_lt_sf_bytes(int64_t nrows, int64_t ncols) {
    const int64_t rows_pad = (nrows + 127) / 128 * 128;
    const int64_t kb_pad = (ncols / 32 + 3) / 4 * 4;
    return (size_t)rows_pad * (size_t)kb_pad;
}

size_t ds4q_mxfp8_lt_bytes(int64_t nrows, int64_t ncols) {
    return (size_t)nrows * (size_t)ncols + ds4q_mxfp8_lt_sf_bytes(nrows, ncols);
}

void ds4q_pack_mxfp8_lt(const uint8_t *fp8_blocks, void *dst,
                        int64_t nrows, int64_t ncols) {
    if (ncols % 32 != 0) {
        fprintf(stderr, "ds4-quantize: mxfp8_lt ncols %lld is not divisible by 32\n",
                (long long)ncols);
        exit(1);
    }
    const int64_t kb_n = ncols / 32;
    /* The LT size equals the type-38 size only for 128-aligned shapes; a
     * non-aligned tensor would grow past its gguf slot and corrupt offsets. */
    if (nrows % 128 != 0 || kb_n % 4 != 0) {
        fprintf(stderr, "ds4-quantize: mxfp8_lt shape [in=%lld,out=%lld] is not "
                "128-aligned (out%%128=%lld, KB%%4=%lld)\n",
                (long long)ncols, (long long)nrows,
                (long long)(nrows % 128), (long long)(kb_n % 4));
        exit(1);
    }
    const size_t data_bytes = (size_t)nrows * (size_t)ncols;
    uint8_t *data = (uint8_t *)dst;
    uint8_t *sf = data + data_bytes;
    const int64_t kb_pad = (kb_n + 3) / 4 * 4;
    memset(sf, 0, ds4q_mxfp8_lt_sf_bytes(nrows, ncols));
    for (int64_t r = 0; r < nrows; r++) {
        for (int64_t kb = 0; kb < kb_n; kb++) {
            const uint8_t *blk = fp8_blocks + ((size_t)r * (size_t)kb_n + (size_t)kb) * 33;
            memcpy(data + (size_t)r * (size_t)ncols + (size_t)kb * 32, blk + 1, 32);
            sf[ds4q_mx_sfoff(r, kb, kb_pad)] = blk[0];
        }
    }
}

/* ---- IQ2_XXS_MMQ (type 43): llama.cpp-MMQ's aligned-SoA layout. ------------
 * A pure permutation of the raw IQ2_XXS (16) block stream: same bytes, same
 * count, same total size, reordered into a d plane and a 64 B-aligned q plane
 * so the MMQ kernels can issue 32-bit loads instead of pairs of 16-bit ones.
 * Byte-identical to gguf-tools/repack_iq2_mmq.py, which is the reference. ---- */
size_t ds4q_iq2_xxs_mmq_bytes(int64_t nblk) {
    return ds4q_pad((size_t)nblk * 2, 64) + (size_t)nblk * 64;
}

void ds4q_pack_iq2_xxs_mmq(const uint8_t *iq2_blocks, void *dst, int64_t nblk) {
    /* Size identity is what lets this be an in-place layout change: every
     * tensor keeps its gguf data offset. It holds only when the d plane pads to
     * nothing. Refuse rather than silently grow the tensor past its slot. */
    if (nblk % 32 != 0) {
        fprintf(stderr, "ds4-quantize: iq2_xxs_mmq nblk %lld is not a multiple of 32 "
                "(would grow past the raw size)\n", (long long)nblk);
        exit(1);
    }
    const size_t dq = ds4q_pad((size_t)nblk * 2, 64);
    uint8_t *out = (uint8_t *)dst;
    memset(out, 0, dq);                       /* d plane, then the alignment pad */
    for (int64_t b = 0; b < nblk; b++)
        memcpy(out + (size_t)b * 2, iq2_blocks + (size_t)b * 66, 2);
    for (int64_t b = 0; b < nblk; b++)
        memcpy(out + dq + (size_t)b * 64, iq2_blocks + (size_t)b * 66 + 2, 64);
}

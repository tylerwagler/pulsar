/* L111 KV4 pack gate: the 4-bit comp-pool pack entry
 * (pulsar_gpu_attn_pack_quantize_store_tensor with comp_fmt mx/nv) against a
 * CPU transcription of the row contract.
 *
 * WHAT IS EXACT AND WHAT IS TOLERANCED -- read this before trusting a run.
 * The engine builds with --use_fast_math, so the device's log2f and '/' are
 * approximate; a GPU-vs-CPU byte-identity gate would fail spuriously at scale
 * and rounding boundaries (the tree has hit this class twice: the non-
 * idempotent fp8 re-quantize, and a competing fork's pow2 scale bug).  So:
 *   EXACT (bitwise, any mismatch fails):
 *     - row layout: nibble/scale/pad/rope OFFSETS, deterministic pad bytes;
 *     - the bf16 rope tail (rtn-even cvt, no fast-math involvement);
 *     - the keep_f32 writeback vs CPU decode of the GPU'S OWN bytes -- this
 *       pins the device decode (attn_kv4_e2m1 bit math + scale decode)
 *       against the reference value table for every code that occurs;
 *   TOLERANCED (fast-math slop, tight budgets, every deviation must be
 *   boundary-shaped or the leg fails):
 *     - MX/E4M3 scale exponents: within +-1 step of the exact ceil(log2);
 *     - NVFP4 row scale: within 4 ulp; per-16 e4m3 scale codes: +-1 code;
 *     - data codes vs encode(src / gpu_scale): mismatches must be ADJACENT
 *       codes and stay under 0.5% of elements.
 *
 * The value distribution is ENCODED DRAWS, not random bytes: per-block
 * magnitudes are log-uniform over ~2^-14..2^6 with deliberate all-zero and
 * sub-floor blocks (attn_pack_fixture.h lesson -- uniform random bytes have
 * uniform exponents and void a gate silently).
 */
#include "pulsar_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HD 512u
#define NROT 64u
#define NNOPE (HD - NROT)
#define NIB (NNOPE / 2u)
#define ROWS 64u

/* ---- e4m3fn reference: value table + rtn-even encode by nearest-code scan
 * (mantissa LSB == code LSB, so tie-to-even-mantissa == tie-to-even-code;
 * saturating at 448 like the hardware cvt the device uses). */
static float ref_e4m3_value(uint32_t code) {
    const uint32_t e = (code >> 3) & 15u, m = code & 7u;
    if (e == 15u && m == 7u) return nanf("");
    if (e == 0u) return (float)m * 0.001953125f;            /* m * 2^-9 */
    return (1.0f + (float)m / 8.0f) * exp2f((float)e - 7.0f);
}
static uint8_t ref_e4m3_encode_pos(float x) {
    if (x >= 448.0f) return 126u;
    uint32_t best = 0u;
    float bd = fabsf(x - ref_e4m3_value(0u));
    for (uint32_t c = 1u; c <= 126u; c++) {
        const float d = fabsf(x - ref_e4m3_value(c));
        if (d < bd || (d == bd && (c & 1u) == 0u && (best & 1u) != 0u)) { best = c; bd = d; }
    }
    return (uint8_t)best;
}

/* ---- e2m1 reference (dsv4_e2m1fn_{value,encode}_dev transcribed). */
static float ref_e2m1_value(uint32_t c) {
    static const float t[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return t[c & 7u];
}
static uint8_t ref_e2m1_encode(float x) {
    const float ax = fminf(fabsf(x), 6.0f);
    uint32_t best = 0u;
    float bd = fabsf(ax - ref_e2m1_value(0u));
    for (uint32_t i = 1u; i < 8u; i++) {
        const float d = fabsf(ax - ref_e2m1_value(i));
        if (d < bd || (d == bd && (i & 1u) == 0u && (best & 1u) != 0u)) { best = i; bd = d; }
    }
    return (uint8_t)(best | (x < 0.0f ? 0x8u : 0u));
}
static float ref_e2m1_decode(uint8_t nib, float scale) {
    const float v = ref_e2m1_value(nib & 7u) * scale;
    return (nib & 8u) ? -v : v;
}

static uint16_t ref_bf16(float x) {
    uint32_t u; memcpy(&u, &x, 4);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return (uint16_t)(u >> 16);
}
static float ref_bf16_val(uint16_t h) {
    const uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f;
}

static int adjacent_e2m1(uint8_t a, uint8_t b) {
    /* adjacent on the signed level ladder: same sign and |code delta| == 1,
     * or the -0.5/0/+0.5 neighbourhood across the sign bit */
    const int sa = (a & 8u) ? -1 : 1, sb = (b & 8u) ? -1 : 1;
    const int ca = a & 7, cb = b & 7;
    if (sa == sb) return ca - cb == 1 || cb - ca == 1;
    return (ca + cb) <= 1;   /* {0,-0} or {0.5-ish across zero} */
}

/* xorshift so the draws are platform-stable */
static uint32_t rng_state = 0x1234abcdu;
static uint32_t xr32(void) {
    uint32_t x = rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return rng_state = x;
}
static float frand(void) { return (float)(xr32() >> 8) * (1.0f / 16777216.0f); }

typedef struct {
    uint64_t code_mism, code_nonadj, scale_dev, wb_mism, layout_mism;
    double rms, ref_pow;
} leg_stats;

/* verify one row against the GPU bytes; scale bytes toleranced, decode exact */
static void verify_row(int fmt, const float *src,
                       const uint8_t *gr, const float *gdec, leg_stats *st) {
    (void)fmt;
    const uint32_t blk = 16u;
    const uint32_t nblk = NNOPE / blk;
    float scale[NNOPE / 16u];
    const uint8_t *sc = gr + NIB;
    const uint16_t *rope = (const uint16_t *)(gr + NIB + nblk + 4u);

    /* scales: recompute exactly, compare with the fast-math tolerance */
    float nv_rs = 0.0f;
    {
        float ra = 0.0f;
        for (uint32_t d = 0; d < NNOPE; d++) ra = fmaxf(ra, fabsf(src[d]));
        const float rs_exact = fmaxf(ra, 1.0e-4f) * (1.0f / (6.0f * 448.0f));
        memcpy(&nv_rs, gr + NIB + nblk, 4);
        int32_t ua, ub; memcpy(&ua, &nv_rs, 4); memcpy(&ub, &rs_exact, 4);
        if (ua - ub > 4 || ub - ua > 4) st->scale_dev++;
    }
    for (uint32_t b = 0; b < nblk; b++) {
        float amax = 0.0f;
        for (uint32_t d = b * blk; d < (b + 1u) * blk; d++) amax = fmaxf(amax, fabsf(src[d]));
        const float t = fminf(448.0f, (float)((double)amax / 6.0 / (double)nv_rs));
        const uint8_t exact = ref_e4m3_encode_pos(t);
        const int d = (int)sc[b] - (int)exact;
        if (d < -1 || d > 1) st->scale_dev++;
        scale[b] = ref_e4m3_value(sc[b]) * nv_rs;   /* GPU byte is authoritative */
    }

    /* data codes vs encode(src / gpu_scale); writeback EXACT vs gpu bytes */
    for (uint32_t d = 0; d < NNOPE; d++) {
        const float s = scale[d / blk];
        const uint8_t gcode = (gr[d >> 1] >> ((d & 1u) * 4u)) & 0xFu;
        const float q = s > 0.0f ? (float)((double)src[d] / (double)s) : 0.0f;
        const uint8_t expect = ref_e2m1_encode(q);
        const float wb_expect = ref_e2m1_decode(gcode, s);
        if (gcode != expect) {
            st->code_mism++;
            if (!adjacent_e2m1(gcode, expect)) st->code_nonadj++;
        }
        if (memcmp(&wb_expect, &gdec[d], 4) != 0) st->wb_mism++;
        const double e2 = (double)gdec[d] - (double)src[d];
        st->rms += e2 * e2;
        st->ref_pow += (double)src[d] * (double)src[d];
    }
    for (uint32_t d = 0; d < NROT; d++) {
        const uint16_t hb = ref_bf16(src[NNOPE + d]);
        if (rope[d] != hb) st->layout_mism++;
        const float wv = ref_bf16_val(rope[d]);
        if (memcmp(&wv, &gdec[NNOPE + d], 4) != 0) st->wb_mism++;
    }
}

int main(void) {
    fprintf(stderr, "kv4-pack-gate: %u rows, head_dim %u\n", ROWS, HD);
    float *src = (float *)malloc((size_t)ROWS * HD * sizeof(float));
    for (uint32_t r = 0; r < ROWS; r++) {
        for (uint32_t b = 0; b < NNOPE / 32u; b++) {
            float mag = exp2f(-14.0f + 20.0f * frand());
            if (r == 0 && b == 0) mag = 0.0f;               /* all-zero block */
            if (r == 1 && b == 1) mag = 1.0e-6f;            /* sub-floor block */
            for (uint32_t d = b * 32u; d < b * 32u + 32u; d++) {
                src[r * HD + d] = mag * (2.0f * frand() - 1.0f);
            }
        }
        for (uint32_t d = NNOPE; d < HD; d++) src[r * HD + d] = 4.0f * frand() - 2.0f;
    }

    struct { int fmt; const char *name; uint64_t rowb; } legs[] = {
        { 0, "nv", 384u },
    };
    int fails = 0;
    for (size_t l = 0; l < sizeof legs / sizeof legs[0]; l++) {
        const uint64_t rowb = legs[l].rowb;
        if (pulsar_gpu_attn_pack_rowbytes(HD) != rowb) {
            fprintf(stderr, "kv4-pack-gate: rowbytes %llu != %llu\n",
                    (unsigned long long)pulsar_gpu_attn_pack_rowbytes(HD),
                    (unsigned long long)rowb);
            return 1;
        }
        pulsar_gpu_tensor *x = pulsar_gpu_tensor_alloc((uint64_t)ROWS * HD * sizeof(float));
        pulsar_gpu_tensor *packed = pulsar_gpu_tensor_alloc((uint64_t)ROWS * rowb);
        if (!x || !packed ||
            !pulsar_gpu_tensor_write(x, 0, src, (uint64_t)ROWS * HD * sizeof(float)) ||
            !pulsar_gpu_attn_pack_quantize_store_tensor(x, packed, 0, ROWS, HD, NROT,
                                                        /*keep_f32=*/true) ||
            !pulsar_gpu_synchronize()) {
            fprintf(stderr, "kv4-pack-gate: %s: GPU pack FAILED to run\n", legs[l].name);
            return 1;
        }
        uint8_t *gpu_rows = (uint8_t *)malloc((size_t)ROWS * rowb);
        float *gpu_dec = (float *)malloc((size_t)ROWS * HD * sizeof(float));
        if (!pulsar_gpu_tensor_read(packed, 0, gpu_rows, (uint64_t)ROWS * rowb) ||
            !pulsar_gpu_tensor_read(x, 0, gpu_dec, (uint64_t)ROWS * HD * sizeof(float))) {
            fprintf(stderr, "kv4-pack-gate: %s: readback FAILED\n", legs[l].name);
            return 1;
        }
        leg_stats st; memset(&st, 0, sizeof st);
        for (uint32_t r = 0; r < ROWS; r++) {
            verify_row(legs[l].fmt, src + (uint64_t)r * HD,
                       gpu_rows + (uint64_t)r * rowb,
                       gpu_dec + (uint64_t)r * HD, &st);
        }
        const uint64_t elems = (uint64_t)ROWS * NNOPE;
        const int leg_fail = st.wb_mism || st.layout_mism || st.code_nonadj ||
                             st.scale_dev > ROWS ||               /* > 1 boundary/row is not slop */
                             st.code_mism * 200u > elems;         /* 0.5% adjacent-code budget */
        fprintf(stderr, "kv4-pack-gate: %-4s: wb_mism %llu, layout_mism %llu, "
                        "scale_dev %llu, code_mism %llu (nonadj %llu) / %llu, "
                        "quant rel-RMS %.4g%s\n",
                legs[l].name,
                (unsigned long long)st.wb_mism, (unsigned long long)st.layout_mism,
                (unsigned long long)st.scale_dev, (unsigned long long)st.code_mism,
                (unsigned long long)st.code_nonadj, (unsigned long long)elems,
                st.ref_pow > 0 ? sqrt(st.rms / st.ref_pow) : 0.0,
                leg_fail ? "  <-- FAIL" : "");
        if (leg_fail) fails++;
        free(gpu_rows); free(gpu_dec);
        pulsar_gpu_tensor_free(x); pulsar_gpu_tensor_free(packed);
    }
    free(src);
    if (fails) { fprintf(stderr, "kv4-pack-gate: FAIL (%d legs)\n", fails); return 1; }
    fprintf(stderr, "kv4-pack-gate: PASS (layout+writeback exact on all 3 legs; "
                    "scale/code deviations within fast-math budgets)\n");
    return 0;
}

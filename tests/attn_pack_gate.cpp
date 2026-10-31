/* Attn-pack gate: 0731's unified NVFP4 row packer
 * (pulsar_gpu_attn_pack_store_tensor, src/cuda/pulsar_cuda_attnpack.cu) against
 * a CPU transcription of the row contract.
 *
 * Restored from dev's tests/kv4_pack_gate.cpp with two adaptations: the entry
 * point is this tree's (`attn_pack_store_tensor`, which takes the f32 source and
 * an optional writeback target rather than dev's in-place + keep_f32 form), and
 * the row geometry is read from the shipped macro rather than dev's
 * pulsar_gpu_attn_pack_rowbytes() accessor -- the macro IS the shipped value, so
 * the gate is still checking what the kernel indexes with and not a
 * transcription of it.
 *
 * WHERE THIS SITS.  tests/attn_pack_fixture_test.cpp is the HOST oracle: it
 * pins the geometry and the recipe arithmetic with no device.  This is the
 * DEVICE gate: it runs the real kernel and compares its bytes and its f32
 * writeback against that contract.  The fixture is not used here on purpose --
 * an oracle shared with the code under test can be wrong in the same direction
 * -- so the reference below is an independent transcription.
 *
 * EVERYTHING IS EXACT (bitwise; any mismatch fails the leg) -- since L292.
 * The TU builds with --use_fast_math, but both of the kernel's divides are
 * __fdiv_rn and its products are single IEEE multiplies, so the device row is
 * a function of IEEE single arithmetic and this gate transcribes it as such
 * (compiled -fno-fast-math).  Before L292 the two divides were the fast-math
 * `/` (MUFU.RCP x FMUL) and this gate tolerated +-1 scale code and 0.5%
 * adjacent data codes; an offline replay of that arithmetic (L292 row) put the
 * drift at ~1e-7 of codes on f32 KV and ~0.03% on bf16-lattice inputs.
 *   - row layout: nibble/scale/pad/rope OFFSETS, deterministic pad bytes;
 *   - the f32 row scale: max(amax, 1e-4) * (1 / 2688), bitwise;
 *   - per-16 e4m3 scale codes: RNE of min(448, (amax * (1/6)) / row_scale);
 *   - data codes: e2m1 RNE of src / scale with the GPU's own decoded scale;
 *   - the bf16 rope tail (rtn-even cvt);
 *   - the f32 writeback vs CPU decode of the GPU'S OWN bytes -- this pins the
 *     device decode (pulsar_e2m1_times bit math + scale decode) against the
 *     reference value table for every code that occurs.
 * (FTZ: the TU flushes subnormals, the host does not; no quantity here comes
 * near 2^-126 -- the row scale is floored at 1e-4 / 2688.)
 *
 * Two legs.  DRAWS: per-block magnitudes log-uniform over ~2^-14..2^6 with
 * deliberate all-zero and sub-floor blocks (attn_pack_fixture.h lesson --
 * uniform random bytes have uniform exponents and void a gate silently).
 * TIES: every non-max element of a block placed ON an E2M1 rounding midpoint
 * of that block's scale (RN(mid * s)) or 1-2 ulp beside it -- the inputs a
 * rounded-reciprocal divide misrounds (the offline replay of the old kernel
 * misses ~5-20% of them).
 *
 * Runs on a GPU.  `make cuda-attn-pack-gate`.  plans/96-two-profiles-one-engine.md s13. */
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
static void verify_row(const float *src, const uint8_t *gr, const float *gdec, leg_stats *st) {
    const uint32_t blk = 16u;
    const uint32_t nblk = NNOPE / blk;
    float scale[NNOPE / 16u];
    const uint8_t *sc = gr + NIB;
    const uint16_t *rope = (const uint16_t *)(gr + NIB + nblk + 4u);

    /* scales: recompute in IEEE single, compare bitwise */
    float nv_rs = 0.0f;
    {
        float ra = 0.0f;
        for (uint32_t d = 0; d < NNOPE; d++) ra = fmaxf(ra, fabsf(src[d]));
        const float rs_exact = fmaxf(ra, 1.0e-4f) * (1.0f / (6.0f * 448.0f));
        memcpy(&nv_rs, gr + NIB + nblk, 4);
        if (memcmp(&nv_rs, &rs_exact, 4) != 0) st->scale_dev++;
    }
    for (uint32_t b = 0; b < nblk; b++) {
        float amax = 0.0f;
        for (uint32_t d = b * blk; d < (b + 1u) * blk; d++) amax = fmaxf(amax, fabsf(src[d]));
        const float t = fminf(448.0f, (amax * (1.0f / 6.0f)) / nv_rs);
        if (sc[b] != ref_e4m3_encode_pos(t)) st->scale_dev++;
        scale[b] = ref_e4m3_value(sc[b]) * nv_rs;   /* the GPU byte decoded: one mismatch counts once */
    }

    /* data codes vs encode(src / gpu_scale) in IEEE single; writeback vs the gpu bytes */
    for (uint32_t d = 0; d < NNOPE; d++) {
        const float s = scale[d / blk];
        const uint8_t gcode = (gr[d >> 1] >> ((d & 1u) * 4u)) & 0xFu;
        const uint8_t expect = s > 0.0f ? ref_e2m1_encode(src[d] / s) : 0u;
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

/* Pack ROWS rows on the GPU (src -> writeback x) and verify every row; 1 on failure. */
static int run_leg(const char *name, const float *src) {
    const uint64_t rowb = PULSAR_ATTN_PACK_ROWBYTES(HD);
    pulsar_gpu_tensor *x = pulsar_gpu_tensor_alloc((uint64_t)ROWS * HD * sizeof(float));
    pulsar_gpu_tensor *packed = pulsar_gpu_tensor_alloc((uint64_t)ROWS * rowb);
    /* src = x and x = the writeback target: the kernel quantises the f32 rows
     * and leaves the DECODED values behind, which is what makes the device
     * decode checkable from the host. */
    if (!x || !packed ||
        !pulsar_gpu_tensor_write(x, 0, src, (uint64_t)ROWS * HD * sizeof(float)) ||
        !pulsar_gpu_attn_pack_store_tensor(x, x, packed, 0, ROWS, HD) ||
        !pulsar_gpu_synchronize()) {
        fprintf(stderr, "attn-pack-gate: %s: GPU pack FAILED to run\n", name);
        return 1;
    }
    uint8_t *gpu_rows = (uint8_t *)malloc((size_t)ROWS * rowb);
    float *gpu_dec = (float *)malloc((size_t)ROWS * HD * sizeof(float));
    if (!pulsar_gpu_tensor_read(packed, 0, gpu_rows, (uint64_t)ROWS * rowb) ||
        !pulsar_gpu_tensor_read(x, 0, gpu_dec, (uint64_t)ROWS * HD * sizeof(float))) {
        fprintf(stderr, "attn-pack-gate: %s: readback FAILED\n", name);
        return 1;
    }
    leg_stats st; memset(&st, 0, sizeof st);
    for (uint32_t r = 0; r < ROWS; r++) {
        verify_row(src + (uint64_t)r * HD, gpu_rows + (uint64_t)r * rowb,
                   gpu_dec + (uint64_t)r * HD, &st);
    }
    const uint64_t elems = (uint64_t)ROWS * NNOPE;
    const int leg_fail = st.wb_mism || st.layout_mism || st.scale_dev || st.code_mism;
    fprintf(stderr, "attn-pack-gate: %s: wb_mism %llu, layout_mism %llu, "
                    "scale_mism %llu, code_mism %llu (nonadj %llu) / %llu, "
                    "quant rel-RMS %.4g%s\n", name,
            (unsigned long long)st.wb_mism, (unsigned long long)st.layout_mism,
            (unsigned long long)st.scale_dev, (unsigned long long)st.code_mism,
            (unsigned long long)st.code_nonadj, (unsigned long long)elems,
            st.ref_pow > 0 ? sqrt(st.rms / st.ref_pow) : 0.0,
            leg_fail ? "  <-- FAIL" : "");
    free(gpu_rows); free(gpu_dec);
    pulsar_gpu_tensor_free(x); pulsar_gpu_tensor_free(packed);
    return leg_fail;
}

int main(void) {
    fprintf(stderr, "attn-pack-gate: %u rows, head_dim %u\n", ROWS, HD);
    /* The gate's own knob must be the shipped one, or it grades a shape the
     * kernel never runs. */
    if (NROT != PULSAR_ATTN_PACK_NROT) {
        fprintf(stderr, "attn-pack-gate: the gate's NROT %u != PULSAR_ATTN_PACK_NROT %u\n",
                NROT, (unsigned)PULSAR_ATTN_PACK_NROT);
        return 1;
    }
    const uint64_t rowb = PULSAR_ATTN_PACK_ROWBYTES(HD);
    if (rowb != 384u) {
        fprintf(stderr, "attn-pack-gate: rowbytes %llu != 384\n", (unsigned long long)rowb);
        return 1;
    }

    float *src = (float *)malloc((size_t)ROWS * HD * sizeof(float));
    int fails = 0;

    /* leg DRAWS */
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
    fails += run_leg("draws", src);

    /* leg TIES: the draws again, then every non-max element of a block moved
     * onto an E2M1 midpoint of that block's scale (this gate's own recipe
     * transcription), 0, 1 or 2 ulp off it.  Each stays below its block's max,
     * so no scale moves. */
    static const float mids[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
    uint64_t placed = 0;
    for (uint32_t r = 0; r < ROWS; r++) {
        float *row = src + (uint64_t)r * HD;
        float ra = 0.0f;
        for (uint32_t d = 0; d < NNOPE; d++) ra = fmaxf(ra, fabsf(row[d]));
        const float rs = fmaxf(ra, 1.0e-4f) * (1.0f / (6.0f * 448.0f));
        for (uint32_t b = 0; b < NNOPE / 16u; b++) {
            uint32_t am = b * 16u;
            for (uint32_t d = b * 16u; d < b * 16u + 16u; d++) if (fabsf(row[d]) > fabsf(row[am])) am = d;
            const float amax = fabsf(row[am]);
            const float t = fminf(448.0f, (amax * (1.0f / 6.0f)) / rs);
            const float s = ref_e4m3_value(ref_e4m3_encode_pos(t)) * rs;
            if (!(s > 0.0f)) continue;
            for (uint32_t d = b * 16u; d < b * 16u + 16u; d++) {
                if (d == am) continue;
                float x = mids[xr32() % 7u] * s;
                const uint32_t off = xr32() % 5u;   /* 0: on it; 1, 3: one ulp; 2, 4: two ulp */
                for (uint32_t k = 0; k < (off + 1u) / 2u; k++) x = nextafterf(x, (off & 1u) ? 0.0f : INFINITY);
                if (!(x < amax)) continue;
                row[d] = (xr32() & 1u) ? -x : x;
                placed++;
            }
        }
    }
    fprintf(stderr, "attn-pack-gate: ties: %llu elements on or beside an E2M1 midpoint\n",
            (unsigned long long)placed);
    if (placed * 2u < (uint64_t)ROWS * NNOPE) {
        fprintf(stderr, "attn-pack-gate: ties: too few elements placed -- the leg is void  <-- FAIL\n");
        fails++;
    }
    fails += run_leg("ties", src);
    free(src);
    if (fails) { fprintf(stderr, "attn-pack-gate: FAIL (%d legs)\n", fails); return 1; }
    fprintf(stderr, "attn-pack-gate: PASS (draws + ties: every byte and the writeback exact)\n");
    return 0;
}

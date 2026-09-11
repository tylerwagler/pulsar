/* L218 KV row pack gate: the two V4.1 row packers (pulsar_gpu_winkv_pack_tensor,
 * pulsar_gpu_mainkv_pack_tensor) against the host replica in kv_row_fixture.h,
 * BYTE FOR BYTE, plus the observer writeback against the host decode of the
 * GPU's own bytes.
 *
 * Exact, not toleranced: the packers derive E8M0 scales from the fp32
 * product's bit fields (the reference's fast_round_scale) and divide with
 * __fdiv_rn, so fast-math cannot move a boundary; E4M3 codes come from the
 * hardware cvt (RNE, saturating) and E2M1 codes from a nearest-code search
 * with ties to even.  The host does the same arithmetic in IEEE fp32.  Any
 * byte that differs is a packer defect or a fixture defect, and either one
 * means the attention gates' oracles (built with the same fixture) are
 * looking at different numbers than the kernel.
 *
 * The value distribution is ENCODED DRAWS, not random bytes: per-block
 * magnitudes are log-uniform over ~2^-14..2^6 with deliberate all-zero,
 * sub-floor, and saturating blocks, plus rows whose ratios land ON the RNE
 * ties (bf16 values over a power-of-two scale) so the tie rule is exercised
 * rather than dodged.
 *
 * The window leg also runs the RING form (raw_cap != 0 with a position /
 * seq_id descriptor): the destination slot must be seq * raw_cap + pos %
 * raw_cap, and a row whose seq_id is outside n_banks must store NOTHING --
 * a wrong slot is a position-dependent wrong answer, not a crash.
 */
/* Self-contained like the attention gates: #includes the shipped packer TU so
 * it drives the REAL kernels, links nothing else, and builds standalone on
 * the GPU box (the dev box has no GPU; sparky is aarch64, so a dev-box binary
 * does not travel). */
#include "../src/cuda/pulsar_cuda_kvrows.cu"
int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    fprintf(stderr, "cuda: %s: %s\n", what, cudaGetErrorString(err));
    return 0;
}
#include "kv_row_fixture.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* the tensor handles the packer entries take, over plain device buffers */
static pulsar_gpu_tensor *dev_alloc(uint64_t bytes) {
    pulsar_gpu_tensor *t = (pulsar_gpu_tensor *)calloc(1, sizeof *t);
    if (!t || cudaMalloc(&t->ptr, bytes) != cudaSuccess) { free(t); return NULL; }
    t->bytes = bytes;
    return t;
}
static void dev_free(pulsar_gpu_tensor *t) { if (t) { cudaFree(t->ptr); free(t); } }
static int dev_write(pulsar_gpu_tensor *t, uint64_t off, const void *src, uint64_t n) {
    return cudaMemcpy((uint8_t *)t->ptr + off, src, n, cudaMemcpyHostToDevice) == cudaSuccess;
}
static int dev_read(const pulsar_gpu_tensor *t, uint64_t off, void *dst, uint64_t n) {
    return cudaMemcpy(dst, (const uint8_t *)t->ptr + off, n, cudaMemcpyDeviceToHost) == cudaSuccess;
}
static int dev_sync(void) { return cudaDeviceSynchronize() == cudaSuccess; }

#define HD 512u
#define ROWS 96u

/* xorshift so the draws are platform-stable */
static uint32_t rng_state = 0x1234abcdu;
static uint32_t xr32(void) {
    uint32_t x = rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return rng_state = x;
}
static float frand(void) { return (float)(xr32() >> 8) * (1.0f / 16777216.0f); }

static void fill_rows(float *src) {
    for (uint32_t r = 0; r < ROWS; r++) {
        for (uint32_t b = 0; b < HD / 16u; b++) {
            float mag = exp2f(-14.0f + 20.0f * frand());
            if (r == 0 && b < 2u) mag = 0.0f;                 /* all-zero blocks (both block sizes) */
            if (r == 1 && b < 2u) mag = 1.0e-6f;              /* sub-floor: window 1e-4, main 6*2^-9 */
            if (r == 2 && b < 2u) mag = 6.0e4f;               /* huge: E8M0 far up, E4M3 scale saturates */
            for (uint32_t d = b * 16u; d < b * 16u + 16u; d++) {
                src[r * HD + d] = mag * (2.0f * frand() - 1.0f);
            }
        }
        if (r >= 3u && r < 8u) {
            /* tie rows: values on the E4M3 / E2M1 half-way points under a
             * power-of-two scale -- k + 1/16 in [1,2) is an E4M3 tie, 0.75 /
             * 1.25 / 2.5 / 5 are E2M1 ties */
            static const float ties[8] = {0.75f, 1.25f, 2.5f, 5.0f, 1.0625f, 1.1875f, 3.5f, 1.75f};
            for (uint32_t d = 0; d < HD; d++) {
                const float t = ties[(d + r) & 7u] * exp2f((float)((int)(r % 3u) - 1));
                src[r * HD + d] = (d & 1u) ? -t : t;
            }
        }
    }
}

static int leg_window_pack(const float *src) {
    const uint64_t rowb = PULSAR_WINKV_ROWBYTES(HD);
    pulsar_gpu_tensor *x = dev_alloc((uint64_t)ROWS * HD * sizeof(float));
    pulsar_gpu_tensor *s = dev_alloc((uint64_t)ROWS * HD * sizeof(float));
    pulsar_gpu_tensor *packed = dev_alloc((uint64_t)ROWS * rowb);
    if (!x || !s || !packed ||
        !dev_write(s, 0, src, (uint64_t)ROWS * HD * sizeof(float)) ||
        !pulsar_gpu_winkv_pack_tensor(x, s, packed, 0, ROWS, HD, NULL, NULL, 1u, 0u) ||
        !dev_sync()) {
        fprintf(stderr, "kv-rows-pack-gate: window: GPU pack FAILED to run\n");
        return 1;
    }
    uint8_t *gpu_rows = (uint8_t *)malloc((size_t)ROWS * rowb);
    float *gpu_dec = (float *)malloc((size_t)ROWS * HD * sizeof(float));
    if (!dev_read(packed, 0, gpu_rows, (uint64_t)ROWS * rowb) ||
        !dev_read(x, 0, gpu_dec, (uint64_t)ROWS * HD * sizeof(float))) {
        fprintf(stderr, "kv-rows-pack-gate: window: readback FAILED\n");
        return 1;
    }
    uint64_t byte_mism = 0, wb_mism = 0, code_hist[4] = {0, 0, 0, 0};
    double rms = 0.0, pw = 0.0;
    uint8_t row[PULSAR_WINKV_ROWBYTES(HD)];
    float dec[HD];
    for (uint32_t r = 0; r < ROWS; r++) {
        host_winkv_pack_row(src + (uint64_t)r * HD, row, dec, HD);
        const uint8_t *gr = gpu_rows + (uint64_t)r * rowb;
        for (uint32_t b = 0; b < rowb; b++) byte_mism += gr[b] != row[b];
        for (uint32_t d = 0; d < HD; d++) {
            /* the writeback must be the host decode of the GPU'S bytes */
            const float wb = host_e4m3_times(gr[d], host_e8m0_scale(gr[HD + d / 32u]));
            wb_mism += memcmp(&wb, &gpu_dec[(uint64_t)r * HD + d], 4) != 0;
            /* the printed quant error is over the RANDOM rows only: rows 0-7 are
             * deliberate edge cases (zero / sub-floor / saturating / ties) */
            const double e = (double)gpu_dec[(uint64_t)r * HD + d] - (double)src[(uint64_t)r * HD + d];
            if (r >= 8u) { rms += e * e; pw += (double)src[(uint64_t)r * HD + d] * (double)src[(uint64_t)r * HD + d]; }
            code_hist[(gr[d] & 0x7Fu) == 0u ? 0 : ((gr[d] & 0x7Fu) == 126u ? 2 : 1)]++;
        }
    }
    fprintf(stderr, "kv-rows-pack-gate: window: %u rows x %llu B, byte_mism %llu, wb_mism %llu, "
                    "codes zero/mid/sat %llu/%llu/%llu, quant rel-RMS %.4g%s\n",
            ROWS, (unsigned long long)rowb, (unsigned long long)byte_mism, (unsigned long long)wb_mism,
            (unsigned long long)code_hist[0], (unsigned long long)code_hist[1], (unsigned long long)code_hist[2],
            pw > 0 ? sqrt(rms / pw) : 0.0, (byte_mism || wb_mism) ? "  <-- FAIL" : "");
    /* a gate whose codes never leave one bucket is measuring a degenerate fixture */
    const int degenerate = code_hist[1] == 0u || code_hist[2] == 0u || code_hist[0] == 0u;
    if (degenerate) fprintf(stderr, "kv-rows-pack-gate: window: DEGENERATE fixture (a code bucket is empty)\n");
    free(gpu_rows); free(gpu_dec);
    dev_free(x); dev_free(s); dev_free(packed);
    return (byte_mism || wb_mism || degenerate) ? 1 : 0;
}

static int leg_window_ring(const float *src) {
    /* 3 banks x raw_cap 8; 16 rows: bank t % 4 (bank 3 is DEAD), pos 100 + t + t/4
     * -- chosen so no two live rows share a slot (two blocks writing one slot
     * would make the expected image order-dependent) */
    const uint32_t n_banks = 3u, raw_cap = 8u, n = 16u;
    const uint64_t rowb = PULSAR_WINKV_ROWBYTES(HD);
    int32_t pos[16], seq[16];
    for (uint32_t t = 0; t < n; t++) { pos[t] = (int32_t)(100u + t + t / 4u); seq[t] = (int32_t)(t % 4u); }
    pulsar_gpu_tensor *s = dev_alloc((uint64_t)n * HD * sizeof(float));
    pulsar_gpu_tensor *ring = dev_alloc((uint64_t)n_banks * raw_cap * rowb);
    pulsar_gpu_tensor *dpos = dev_alloc(n * sizeof(int32_t));
    pulsar_gpu_tensor *dseq = dev_alloc(n * sizeof(int32_t));
    uint8_t *fill = (uint8_t *)malloc((size_t)n_banks * raw_cap * rowb);
    memset(fill, 0xA5, (size_t)n_banks * raw_cap * rowb);   /* sentinel: untouched slots keep it */
    if (!s || !ring || !dpos || !dseq ||
        !dev_write(s, 0, src, (uint64_t)n * HD * sizeof(float)) ||
        !dev_write(ring, 0, fill, (uint64_t)n_banks * raw_cap * rowb) ||
        !dev_write(dpos, 0, pos, n * sizeof(int32_t)) ||
        !dev_write(dseq, 0, seq, n * sizeof(int32_t)) ||
        !pulsar_gpu_winkv_pack_tensor(NULL, s, ring, 0u, n, HD, dpos, dseq, n_banks, raw_cap) ||
        !dev_sync()) {
        fprintf(stderr, "kv-rows-pack-gate: ring: GPU pack FAILED to run\n");
        return 1;
    }
    uint8_t *got = (uint8_t *)malloc((size_t)n_banks * raw_cap * rowb);
    if (!dev_read(ring, 0, got, (uint64_t)n_banks * raw_cap * rowb)) {
        fprintf(stderr, "kv-rows-pack-gate: ring: readback FAILED\n");
        return 1;
    }
    /* expected ring: replay the slot rule on the host; later rows win a slot */
    uint8_t *want = fill;
    uint32_t stored = 0u, dead = 0u;
    for (uint32_t t = 0; t < n; t++) {
        if ((uint32_t)seq[t] >= n_banks) { dead++; continue; }
        const uint64_t slot = (uint64_t)seq[t] * raw_cap + (uint32_t)pos[t] % raw_cap;
        host_winkv_pack_row(src + (uint64_t)t * HD, want + slot * rowb, NULL, HD);
        stored++;
    }
    uint64_t mism = 0;
    for (uint64_t b = 0; b < (uint64_t)n_banks * raw_cap * rowb; b++) mism += got[b] != want[b];
    fprintf(stderr, "kv-rows-pack-gate: ring: %u rows -> %u stored, %u dead, %u banks x %u slots: "
                    "byte_mism %llu%s\n", n, stored, dead, n_banks, raw_cap,
            (unsigned long long)mism, mism ? "  <-- FAIL" : "");
    free(fill); free(got);
    dev_free(s); dev_free(ring);
    dev_free(dpos); dev_free(dseq);
    return mism ? 1 : 0;
}

static int leg_main_pack(const float *src) {
    const uint64_t rowb = PULSAR_MAINKV_ROWBYTES(HD);
    pulsar_gpu_tensor *x = dev_alloc((uint64_t)ROWS * HD * sizeof(float));
    pulsar_gpu_tensor *s = dev_alloc((uint64_t)ROWS * HD * sizeof(float));
    pulsar_gpu_tensor *packed = dev_alloc((uint64_t)(ROWS + 5u) * rowb);
    /* out_row0 = 5: the destination offset is part of the contract */
    if (!x || !s || !packed ||
        !dev_write(s, 0, src, (uint64_t)ROWS * HD * sizeof(float)) ||
        !pulsar_gpu_mainkv_pack_tensor(x, s, packed, 5u, ROWS, HD) ||
        !dev_sync()) {
        fprintf(stderr, "kv-rows-pack-gate: main: GPU pack FAILED to run\n");
        return 1;
    }
    uint8_t *gpu_rows = (uint8_t *)malloc((size_t)ROWS * rowb);
    float *gpu_dec = (float *)malloc((size_t)ROWS * HD * sizeof(float));
    if (!dev_read(packed, 5u * rowb, gpu_rows, (uint64_t)ROWS * rowb) ||
        !dev_read(x, 0, gpu_dec, (uint64_t)ROWS * HD * sizeof(float))) {
        fprintf(stderr, "kv-rows-pack-gate: main: readback FAILED\n");
        return 1;
    }
    uint64_t byte_mism = 0, wb_mism = 0, nib_hist[8] = {0};
    double rms = 0.0, pw = 0.0;
    uint8_t row[PULSAR_MAINKV_ROWBYTES(HD)];
    float dec[HD];
    for (uint32_t r = 0; r < ROWS; r++) {
        host_mainkv_pack_row(src + (uint64_t)r * HD, row, dec, HD);
        const uint8_t *gr = gpu_rows + (uint64_t)r * rowb;
        for (uint32_t b = 0; b < rowb; b++) byte_mism += gr[b] != row[b];
        for (uint32_t d = 0; d < HD; d++) {
            const uint8_t nib = (gr[d >> 1] >> ((d & 1u) * 4u)) & 0xFu;
            const float wb = host_e2m1_times(nib, host_e4m3_mag(gr[HD / 2u + d / 16u]));
            wb_mism += memcmp(&wb, &gpu_dec[(uint64_t)r * HD + d], 4) != 0;
            /* the printed quant error is over the RANDOM rows only: rows 0-7 are
             * deliberate edge cases (zero / sub-floor / saturating / ties) */
            const double e = (double)gpu_dec[(uint64_t)r * HD + d] - (double)src[(uint64_t)r * HD + d];
            if (r >= 8u) { rms += e * e; pw += (double)src[(uint64_t)r * HD + d] * (double)src[(uint64_t)r * HD + d]; }
            nib_hist[nib & 7u]++;
        }
    }
    int degenerate = 0;
    for (int c = 0; c < 8; c++) degenerate |= nib_hist[c] == 0u;
    fprintf(stderr, "kv-rows-pack-gate: main:   %u rows x %llu B, byte_mism %llu, wb_mism %llu, "
                    "nibble codes %llu %llu %llu %llu %llu %llu %llu %llu, quant rel-RMS %.4g%s\n",
            ROWS, (unsigned long long)rowb, (unsigned long long)byte_mism, (unsigned long long)wb_mism,
            (unsigned long long)nib_hist[0], (unsigned long long)nib_hist[1], (unsigned long long)nib_hist[2],
            (unsigned long long)nib_hist[3], (unsigned long long)nib_hist[4], (unsigned long long)nib_hist[5],
            (unsigned long long)nib_hist[6], (unsigned long long)nib_hist[7],
            pw > 0 ? sqrt(rms / pw) : 0.0, (byte_mism || wb_mism) ? "  <-- FAIL" : "");
    if (degenerate) fprintf(stderr, "kv-rows-pack-gate: main: DEGENERATE fixture (an E2M1 code never occurs)\n");
    free(gpu_rows); free(gpu_dec);
    dev_free(x); dev_free(s); dev_free(packed);
    return (byte_mism || wb_mism || degenerate) ? 1 : 0;
}

int main(void) {
    fprintf(stderr, "kv-rows-pack-gate: %u rows, head_dim %u (window %llu B, main %llu B)\n", ROWS, HD,
            (unsigned long long)PULSAR_WINKV_ROWBYTES(HD), (unsigned long long)PULSAR_MAINKV_ROWBYTES(HD));
    float *src = (float *)malloc((size_t)ROWS * HD * sizeof(float));
    fill_rows(src);
    int fails = 0;
    fails += leg_window_pack(src);
    fails += leg_window_ring(src);
    fails += leg_main_pack(src);
    free(src);
    if (fails) { fprintf(stderr, "kv-rows-pack-gate: FAIL (%d legs)\n", fails); return 1; }
    fprintf(stderr, "kv-rows-pack-gate: PASS (both row formats byte-exact vs the host replica; "
                    "writeback exact; ring slots exact)\n");
    return 0;
}

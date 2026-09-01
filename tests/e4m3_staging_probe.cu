/* Round-trip ds4_quantize_mmq_e4m3_cuda against its own layout.
 *
 * The MXFP8 expert arm produces garbage and the MMA has been cleared by
 * tests/mxfp8_mma_probe.cu, so the remaining suspects are the pieces written
 * from scratch.  This is the one with the most index math: it must reproduce
 * quantize_mmq_q8_1's block placement exactly, because the D2R staging,
 * prefetch and load_B_tile all assume that placement.
 *
 * The check is the definition of the format: for row r and column k,
 *     dequant = e4m3(qs[ib][k % 128]) * 2^(d4[ib][(k % 128) / 32] - 127)
 * with ib = (k / 128) * rows + r, must reproduce x[r][k] to within e4m3's
 * resolution at that block's shared exponent.
 *
 * Build (or `make tests/e4m3_staging_probe`):
 *   nvcc -O3 -arch=sm_120f -Isrc -Isrc/cuda/mmq -o tests/e4m3_staging_probe \
 *        tests/e4m3_staging_probe.cu
 *
 * Tracks two renames of 2026-08-18: the staging block is block_mx_act_mmq (it
 * holds E4M3 bit patterns, not int8 q8_1 magnitudes -- the old name was being
 * misread as evidence of the latter) and its width is DS4_ACT_BLOCK_VALS.  This
 * probe did not follow them and stopped compiling; nothing noticed, because
 * nothing built it.  It has a Makefile target now.
 */

#include "ds4_quantize_e4m3.cu"

#include <cstdio>
#include <cmath>
#include <cstdlib>

int main() {
    const int rows = 4;
    const int K    = 256;          /* two 128-value blocks per row */
    const int nblk = (K / DS4_ACT_BLOCK_VALS) * rows;

    float *x = nullptr;
    block_mx_act_mmq *y = nullptr;
    cudaMallocManaged(&x, (size_t)rows * K * sizeof(float));
    cudaMallocManaged(&y, (size_t)nblk * sizeof(block_mx_act_mmq));
    cudaMemset(y, 0, (size_t)nblk * sizeof(block_mx_act_mmq));

    /* A distribution with real dynamic range: each 32-block gets its own
     * magnitude so a wrong shared exponent shows up immediately. */
    srand(1234);
    for (int r = 0; r < rows; ++r) {
        for (int k = 0; k < K; ++k) {
            const float mag = ldexpf(1.0f, (k / 32) % 5 - 2);
            x[r * K + k] = mag * ((float)rand() / RAND_MAX * 2.0f - 1.0f);
        }
    }

    ds4_quantize_mmq_e4m3_cuda(
        x, /*ids=*/nullptr, (void *)y,
        /*ne00=*/K, /*s01=*/K, /*s02=*/0, /*s03=*/0,
        /*ne0=*/K, /*ne1=*/rows, /*ne2=*/1, /*ne3=*/1,
        /*n_expert_used=*/0, /*scatter=*/false, /*stream=*/0);
    cudaDeviceSynchronize();
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("LAUNCH FAILED: %s\n", cudaGetErrorString(err));
        return 1;
    }

    double worst = 0.0;
    int wr = -1, wk = -1;
    int zero_scales = 0;
    for (int r = 0; r < rows; ++r) {
        for (int k = 0; k < K; ++k) {
            const int ib   = (k / DS4_ACT_BLOCK_VALS) * rows + r;
            const int iqs  = k % DS4_ACT_BLOCK_VALS;
            const unsigned char qb = (unsigned char)y[ib].qs[iqs];
            const int sb = (int)y[ib].d4[iqs / 32];
            if (sb == 0) { zero_scales++; continue; }
            __nv_fp8_e4m3 e;
            *(unsigned char *)&e = qb;
            const double got  = (double)(float)e * exp2((double)(sb - 127));
            const double want = x[r * K + k];
            const double den  = fabs(want) > 1e-8 ? fabs(want) : 1.0;
            const double rel  = fabs(got - want) / den;
            if (rel > worst) { worst = rel; wr = r; wk = k; }
        }
    }
    printf("e4m3 staging round-trip: worst rel err %.4f at (row=%d, k=%d)\n",
           worst, wr, wk);
    printf("  zero scale bytes seen: %d (expect 0)\n", zero_scales);
    if (wr >= 0) {
        const int ib  = (wk / DS4_ACT_BLOCK_VALS) * rows + wr;
        const int iqs = wk % DS4_ACT_BLOCK_VALS;
        printf("  worst: want %.6f  scale_byte=%d  qs=0x%02x\n",
               x[wr * K + wk], (int)y[ib].d4[iqs / 32],
               (unsigned char)y[ib].qs[iqs]);
    }
    /* e4m3 carries 3 mantissa bits, so a correctly scaled block should land
     * inside ~6-7% relative; anything far above that is a layout bug, not
     * quantisation. */
    printf("%s\n", worst < 0.10 ? "STAGING ROUND-TRIP: PLAUSIBLE"
                                : "STAGING ROUND-TRIP: BROKEN");
    return 0;
}

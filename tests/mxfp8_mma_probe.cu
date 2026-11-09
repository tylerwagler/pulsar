/* Isolate the block-scaled MXFP8 MMA's operand and SCALE conventions.
 *
 * The engine-level arm (PULSAR_MOE_IQ2_ARM=1) produces token soup, and an
 * engine cycle is ~7 minutes and exercises everything at once.  This runs the
 * instruction alone against a CPU reference in seconds, exactly as
 * tests/attn_mma_probe.cu does for the attention fragment layout -- whose own
 * header records that it caught two silent-wrong-answer layout traps.
 *
 * PROBES, each isolating one thing:
 *   1. all-ones, unit scales      -> is the basic contraction right? (expect 32)
 *   2. A row-ramp                 -> which rows does a lane's a0..a3 feed?
 *   3. B col-ramp                 -> which cols does a lane's b0/b1 feed?
 *   4. A per-row scale ramp       -> the sfa lane convention
 *   5. B per-col scale ramp       -> the sfb lane convention
 *
 * Build:
 *   nvcc -O3 -arch=sm_120f -Isrc -Isrc/cuda/mmq -o tests/mxfp8_mma_probe \
 *        tests/mxfp8_mma_probe.cu
 */

#include "ds4_mxfp8_mma.cuh"

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#define M 16
#define N 8
#define K 32

/* Fragment layout for the 8-bit m16n8k32 shape (same as .s8.s8, which the
 * integer D2R path already relies on):
 *   A: lane L, reg a0 -> row L/4,     k = (L%4)*4 + 0..3
 *              reg a1 -> row L/4 + 8, k = (L%4)*4 + 0..3
 *              reg a2 -> row L/4,     k = 16 + (L%4)*4 + 0..3
 *              reg a3 -> row L/4 + 8, k = 16 + (L%4)*4 + 0..3
 *   B: lane L, reg b0 -> col L/4,     k = (L%4)*4 + 0..3
 *              reg b1 -> col L/4,     k = 16 + (L%4)*4 + 0..3
 *   C: lane L, elem l -> row (l<2 ? 0 : 8) + L/4, col (l%2) + 2*(L%4)
 */
__global__ static void probe_kernel(const float *A, const float *B,
                                    const unsigned char *sfa_row,
                                    const unsigned char *sfb_col,
                                    float *out, int sfa_mode, int sfb_mode) {
    const int lane = threadIdx.x;
    const int g    = lane >> 2;      /* 0..7 */
    const int tg   = lane & 3;       /* 0..3 */

    unsigned char a[16], b[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int k_lo = tg * 4 + j;
        const int k_hi = 16 + tg * 4 + j;
        a[0 * 4 + j] = (unsigned char)__nv_cvt_float_to_fp8(A[(g)     * K + k_lo], __NV_SATFINITE, __NV_E4M3);
        a[1 * 4 + j] = (unsigned char)__nv_cvt_float_to_fp8(A[(g + 8) * K + k_lo], __NV_SATFINITE, __NV_E4M3);
        a[2 * 4 + j] = (unsigned char)__nv_cvt_float_to_fp8(A[(g)     * K + k_hi], __NV_SATFINITE, __NV_E4M3);
        a[3 * 4 + j] = (unsigned char)__nv_cvt_float_to_fp8(A[(g + 8) * K + k_hi], __NV_SATFINITE, __NV_E4M3);
        b[0 * 4 + j] = (unsigned char)__nv_cvt_float_to_fp8(B[g * K + k_lo], __NV_SATFINITE, __NV_E4M3);
        b[1 * 4 + j] = (unsigned char)__nv_cvt_float_to_fp8(B[g * K + k_hi], __NV_SATFINITE, __NV_E4M3);
    }
    uint32_t a0 = *(const uint32_t *)(a + 0),  a1 = *(const uint32_t *)(a + 4);
    uint32_t a2 = *(const uint32_t *)(a + 8),  a3 = *(const uint32_t *)(a + 12);
    uint32_t b0 = *(const uint32_t *)(b + 0),  b1 = *(const uint32_t *)(b + 4);

    /* Candidate sfa conventions.  Mode 0 is the one copied from
     * pulsar_cuda_indexer_mxfp4.cu (tig==1 supplies row g+8). */
    uint32_t sfa = 0, sfb = 0;
    switch (sfa_mode) {
        case 0: sfa = sfa_row[(tg == 1) ? (g + 8) : g]; break;
        case 1: sfa = sfa_row[g] | ((uint32_t)sfa_row[g + 8] << 8); break;
        case 2: sfa = (tg < 2) ? sfa_row[g] : sfa_row[g + 8]; break;
        default: sfa = sfa_row[g]; break;
    }
    switch (sfb_mode) {
        case 0: sfb = sfb_col[g]; break;
        case 1: sfb = sfb_col[g] | ((uint32_t)sfb_col[g] << 8); break;
        default: sfb = sfb_col[g]; break;
    }

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    ds4_mma_m16n8k32_e4m3(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, sfa, sfb);

    const float dv[4] = {d0, d1, d2, d3};
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const int row = ((l < 2) ? 0 : 8) + g;
        const int col = (l & 1) + 2 * tg;
        out[row * N + col] = dv[l];
    }
}

static void ref_gemm(const float *A, const float *B, const unsigned char *sfa,
                     const unsigned char *sfb, float *R) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                /* values are round-tripped through e4m3 exactly as the kernel does */
                const float av = (float)(__nv_fp8_e4m3)A[m * K + k];
                const float bv = (float)(__nv_fp8_e4m3)B[n * K + k];
                acc += (double)av * (double)bv;
            }
            acc *= exp2((double)((int)sfa[m] - 127)) * exp2((double)((int)sfb[n] - 127));
            R[m * N + n] = (float)acc;
        }
    }
}

int main() {
    printf("DS4_MXFP8_MMA compiled-in (host view) = %d\n", (int)DS4_MXFP8_MMA);

    float *A, *B, *out;
    unsigned char *sfa, *sfb;
    cudaMallocManaged(&A, M * K * sizeof(float));
    cudaMallocManaged(&B, N * K * sizeof(float));
    cudaMallocManaged(&out, M * N * sizeof(float));
    cudaMallocManaged(&sfa, M);
    cudaMallocManaged(&sfb, N);
    float R[M * N];

    const char *names[5] = {"1: all-ones, unit scales", "2: A row-ramp",
                            "3: B col-ramp", "4: A row-SCALE ramp",
                            "5: B col-SCALE ramp"};

    for (int probe = 0; probe < 5; ++probe) {
        for (int m = 0; m < M; ++m) for (int k = 0; k < K; ++k) A[m * K + k] = 1.0f;
        for (int n = 0; n < N; ++n) for (int k = 0; k < K; ++k) B[n * K + k] = 1.0f;
        for (int m = 0; m < M; ++m) sfa[m] = 127;
        for (int n = 0; n < N; ++n) sfb[n] = 127;
        if (probe == 1) for (int m = 0; m < M; ++m) for (int k = 0; k < K; ++k) A[m * K + k] = (float)(m + 1);
        if (probe == 2) for (int n = 0; n < N; ++n) for (int k = 0; k < K; ++k) B[n * K + k] = (float)(n + 1);
        /* MUST distinguish row m from row m+8, or every sfa convention ties and
         * the probe proves nothing: (m % 4) gives rows m and m+8 the SAME scale
         * because 8 % 4 == 0.  (m % 3) does not (8 % 3 == 2). */
        if (probe == 3) for (int m = 0; m < M; ++m) sfa[m] = (unsigned char)(127 + (m % 3));
        if (probe == 4) for (int n = 0; n < N; ++n) sfb[n] = (unsigned char)(127 + (n % 3));

        for (int sm = 0; sm < 4; ++sm) {
            for (int m = 0; m < M * N; ++m) out[m] = -1.0f;
            probe_kernel<<<1, 32>>>(A, B, sfa, sfb, out, sm, 0);
            cudaDeviceSynchronize();
            ref_gemm(A, B, sfa, sfb, R);
            double worst = 0.0; int wm = -1, wn = -1;
            for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
                const double den = fabs(R[m * N + n]) > 1e-6 ? fabs(R[m * N + n]) : 1.0;
                const double e = fabs(out[m * N + n] - R[m * N + n]) / den;
                if (e > worst) { worst = e; wm = m; wn = n; }
            }
            printf("probe %-24s sfa_mode=%d  worst rel err %.3e", names[probe], sm, worst);
            if (worst > 1e-5) printf("   at (m=%d,n=%d) got %.4f want %.4f", wm, wn,
                                     out[wm * N + wn], R[wm * N + wn]);
            printf("\n");
            if (probe == 0 && sm == 0) {
                printf("   out[0][0..3] = %.3f %.3f %.3f %.3f  (expect 32 each)\n",
                       out[0], out[1], out[2], out[3]);
            }
        }
    }
    return 0;
}

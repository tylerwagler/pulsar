/* Verify the m16n8k16 f16 fragment layout BEFORE building an attention kernel
 * on top of it.
 *
 * The block-scaled work (tests/idx_mxfp4_probe.cu) turned up two contract
 * details that were silent-wrong-answer traps rather than errors -- the ue8m0
 * bias was 128 and not the spec's 127, and the e2m1 nibble sat at bits[5:2]
 * where a bit-0 reading agrees for codes 0..4 and diverges at 5,6,7.  A layout
 * mistake here would behave the same way: plausible attention output, wrong
 * values, no diagnostic.  So the layout is asserted against a CPU reference on
 * random data before any kernel depends on it.
 *
 * Hypothesis under test (PTX ISA, mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32),
 * with g = lane>>2 and t = lane&3:
 *   A 16x16 row-major, 4 regs = 8 halves:
 *     r0 = {A[g][2t],   A[g][2t+1]}     r1 = {A[g+8][2t],   A[g+8][2t+1]}
 *     r2 = {A[g][2t+8], A[g][2t+9]}     r3 = {A[g+8][2t+8], A[g+8][2t+9]}
 *   B 16x8 (K x N), 2 regs = 4 halves:
 *     r0 = {B[2t][g],   B[2t+1][g]}     r1 = {B[2t+8][g],   B[2t+9][g]}
 *   D 16x8 f32, 4 regs:
 *     d0 = D[g][2t]     d1 = D[g][2t+1]  d2 = D[g+8][2t]  d3 = D[g+8][2t+1]
 *
 * build+run:
 *   nvcc -O3 -arch=sm_120f -o /tmp/attnprobe tests/attn_mma_probe.cu
 *   /tmp/attnprobe
 */
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define M 16
#define N 8
#define K 16

__global__ static void probe_kernel(const __half *A, const __half *B, float *D) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t g = lane >> 2u, t = lane & 3u;

    /* A: row-major [M][K] */
    __half2 a0 = make_half2(A[(g)     * K + 2*t],     A[(g)     * K + 2*t + 1]);
    __half2 a1 = make_half2(A[(g + 8) * K + 2*t],     A[(g + 8) * K + 2*t + 1]);
    __half2 a2 = make_half2(A[(g)     * K + 2*t + 8], A[(g)     * K + 2*t + 9]);
    __half2 a3 = make_half2(A[(g + 8) * K + 2*t + 8], A[(g + 8) * K + 2*t + 9]);
    /* B: [K][N], the MMA reads it as col-major (n is the outer index) */
    __half2 b0 = make_half2(B[(2*t)     * N + g], B[(2*t + 1) * N + g]);
    __half2 b1 = make_half2(B[(2*t + 8) * N + g], B[(2*t + 9) * N + g]);

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(*(const uint32_t *)&a0), "r"(*(const uint32_t *)&a1),
          "r"(*(const uint32_t *)&a2), "r"(*(const uint32_t *)&a3),
          "r"(*(const uint32_t *)&b0), "r"(*(const uint32_t *)&b1));

    D[(g)     * N + 2*t]     = d0;
    D[(g)     * N + 2*t + 1] = d1;
    D[(g + 8) * N + 2*t]     = d2;
    D[(g + 8) * N + 2*t + 1] = d3;
}

int main(void) {
    __half hA[M * K], hB[K * N];
    float ref[M * N], got[M * N];
    /* Values exactly representable in f16 so the reference is exact and any
     * mismatch is a LAYOUT bug, not rounding. */
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++) hA[m * K + k] = __float2half((float)((m * 7 + k * 3) % 13 - 6));
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++) hB[k * N + n] = __float2half((float)((k * 5 + n * 11) % 11 - 5));
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float acc = 0.f;
            for (int k = 0; k < K; k++)
                acc += __half2float(hA[m * K + k]) * __half2float(hB[k * N + n]);
            ref[m * N + n] = acc;
        }

    __half *dA, *dB; float *dD;
    cudaMalloc(&dA, sizeof(hA)); cudaMalloc(&dB, sizeof(hB)); cudaMalloc(&dD, sizeof(got));
    cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice);
    probe_kernel<<<1, 32>>>(dA, dB, dD);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        printf("EXEC FAILED: %s\nPROBE: FAIL\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    cudaMemcpy(got, dD, sizeof(got), cudaMemcpyDeviceToHost);

    int bad = 0;
    for (int i = 0; i < M * N; i++) if (ref[i] != got[i]) bad++;
    printf("m16n8k16 f16 layout probe: %d/%d elements mismatched\n", bad, M * N);
    if (bad) {
        printf("  first rows (ref | got):\n");
        for (int m = 0; m < 4; m++) {
            printf("   m=%2d ", m);
            for (int n = 0; n < N; n++) printf("%7.0f", ref[m * N + n]);
            printf("   |");
            for (int n = 0; n < N; n++) printf("%7.0f", got[m * N + n]);
            printf("\n");
        }
    }
    printf("PROBE: %s\n", bad ? "FAIL" : "PASS -- layout hypothesis confirmed");
    return bad ? 1 : 0;
}

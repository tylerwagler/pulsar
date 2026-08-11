/* Does --use_fast_math fold the fp16 round-trip back to identity?
 *
 * The Q/P attention emulation rounds via __half2float(__float2half(v)). In the
 * engine that produced NO change at all (bit-identical dumps), while the bf16
 * variant rounded normally. This isolates the question with the same compile
 * flags and no model: if `naive` comes back equal to its input for every value
 * whose f32 mantissa cannot fit in fp16's 10 bits, the optimizer folded it.
 *
 * Build (same flags as the engine's cuda objects):
 *   nvcc -O3 --use_fast_math -arch=sm_120f tests/fp16_fold_probe.cu -o fp16_probe
 */
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

__global__ void probe(const float *in, float *naive, float *bitcast,
                      float *bf16, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = in[i];

    /* what the emulation currently does */
    naive[i] = __half2float(__float2half(v));

    /* force the 16-bit pattern to be materialised through an integer */
    __half h = __float2half_rn(v);
    unsigned short bits = __half_as_ushort(h);
    bitcast[i] = __half2float(__ushort_as_half(bits));

    /* the bf16 path, which demonstrably DOES round inside the engine */
    bf16[i] = __bfloat162float(__float2bfloat16(v));
}

int main(void) {
    const int n = 8;
    /* values whose f32 mantissa cannot survive 10 bits of fp16 */
    float h_in[n] = {1.0000001f, 3.14159265f, 0.30000001192f, 19.42f,
                     0.123456789f, 1.9999999f, 7.7777777f, 0.00012345678f};
    float *d_in, *d_a, *d_b, *d_c;
    cudaMalloc(&d_in, n * 4); cudaMalloc(&d_a, n * 4);
    cudaMalloc(&d_b, n * 4);  cudaMalloc(&d_c, n * 4);
    cudaMemcpy(d_in, h_in, n * 4, cudaMemcpyHostToDevice);
    probe<<<1, n>>>(d_in, d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    float a[n], b[n], c[n];
    cudaMemcpy(a, d_a, n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(b, d_b, n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(c, d_c, n * 4, cudaMemcpyDeviceToHost);

    int naive_noop = 0, bitcast_noop = 0, bf16_noop = 0;
    printf("%-16s %-16s %-16s %-16s\n", "input", "naive fp16", "bitcast fp16", "bf16");
    for (int i = 0; i < n; i++) {
        if (a[i] == h_in[i]) naive_noop++;
        if (b[i] == h_in[i]) bitcast_noop++;
        if (c[i] == h_in[i]) bf16_noop++;
        printf("%-16.9g %-16.9g %-16.9g %-16.9g\n", h_in[i], a[i], b[i], c[i]);
    }
    printf("\nunchanged-by-rounding counts (want 0):\n");
    printf("  naive fp16   : %d/%d %s\n", naive_noop, n,
           naive_noop == n ? "<-- FOLDED TO IDENTITY" : "");
    printf("  bitcast fp16 : %d/%d\n", bitcast_noop, n);
    printf("  bf16         : %d/%d\n", bf16_noop, n);
    return 0;
}

/* L112 inc C: is ANY fixed cuBLAS algorithm M-stable for the bf16 core's
 * GemmEx shape family?  Replicates matmul_bf16_wptr's exact call:
 *   GemmEx(OP_T, OP_N, out_dim, n_tok, in_dim, w bf16 lda=in_dim,
 *          x bf16 ldb=in_dim, out f32 ldc=out_dim, compute F32, ALGO)
 * For each algo: run the SAME device buffers at n_tok=M1 and n_tok=M2 and
 * bitwise-compare the first M1 output columns.  M-STABLE means a row's
 * result does not depend on how many other rows ride the call -- the
 * property the co-batch requirement needs at tensor-core speed.
 *
 *   usage: ./tests/cublas_mstable_probe
 */
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static void fill_bf16(uint16_t *dst_dev, uint64_t n, uint32_t seed) {
    uint16_t *h = (uint16_t *)malloc(n * sizeof(uint16_t));
    uint32_t s = seed;
    for (uint64_t i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        /* map to a sane bf16 range: exponent around 1.0, varied mantissa/sign */
        float f = ((float)(s >> 8) / (float)(1u << 24) - 0.5f) * 2.0f;
        uint32_t u; memcpy(&u, &f, 4);
        h[i] = (uint16_t)(u >> 16);
    }
    cudaMemcpy(dst_dev, h, n * sizeof(uint16_t), cudaMemcpyHostToDevice);
    free(h);
}

int main(void) {
    cublasHandle_t bl;
    if (cublasCreate(&bl) != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublas create failed\n"); return 1; }

    struct { int K, N; const char *tag; } shapes[] = {
        { 16384,   24, "hc_mix (census real)" },
        {  4096,   64, "census real 64" },
        {  4096,  256, "census real 256" },
        {  4096,  512, "census real 512" },
        {  4096, 1024, "census real 1024" },
    };
    const int M2 = 514, M1 = 257;
    const int MB = 4096;   /* also test 2048-vs-4096 for the big-chunk case */

    for (unsigned si = 0; si < sizeof(shapes)/sizeof(shapes[0]); si++) {
        const int K = shapes[si].K, N = shapes[si].N;
        uint16_t *w, *x; float *o1, *o2;
        cudaMalloc(&w, (uint64_t)N * K * 2);
        cudaMalloc(&x, (uint64_t)MB * K * 2);
        cudaMalloc(&o1, (uint64_t)MB * N * 4);
        cudaMalloc(&o2, (uint64_t)MB * N * 4);
        fill_bf16(w, (uint64_t)N * K, 0xa123u + si);
        fill_bf16(x, (uint64_t)MB * K, 0xa987u + si);
        float *h1 = (float *)malloc((uint64_t)MB * N * 4);
        float *h2 = (float *)malloc((uint64_t)MB * N * 4);

        printf("shape K=%d N=%d (%s)\n", K, N, shapes[si].tag);
        for (int algo = -1; algo <= 23; algo++) {
            const float alpha = 1.0f, beta = 0.0f;
            cudaMemset(o1, 0, (uint64_t)MB * N * 4);
            cudaMemset(o2, 0, (uint64_t)MB * N * 4);
            cublasStatus_t s1 = cublasGemmEx(bl, CUBLAS_OP_T, CUBLAS_OP_N,
                    N, M1, K, &alpha, w, CUDA_R_16BF, K, x, CUDA_R_16BF, K,
                    &beta, o1, CUDA_R_32F, N, CUDA_R_32F, (cublasGemmAlgo_t)algo);
            cublasStatus_t s2 = cublasGemmEx(bl, CUBLAS_OP_T, CUBLAS_OP_N,
                    N, M2, K, &alpha, w, CUDA_R_16BF, K, x, CUDA_R_16BF, K,
                    &beta, o2, CUDA_R_32F, N, CUDA_R_32F, (cublasGemmAlgo_t)algo);
            if (s1 != CUBLAS_STATUS_SUCCESS || s2 != CUBLAS_STATUS_SUCCESS) {
                printf("  algo %3d: unsupported (%d/%d)\n", algo, (int)s1, (int)s2);
                continue;
            }
            cudaDeviceSynchronize();
            cudaMemcpy(h1, o1, (uint64_t)M1 * N * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(h2, o2, (uint64_t)M1 * N * 4, cudaMemcpyDeviceToHost);
            long fd = -1;
            for (long i = 0; i < (long)M1 * N; i++) if (h1[i] != h2[i]) { fd = i; break; }
            /* second pair: 2048 vs 4096 (big-chunk regime) */
            cublasStatus_t s3 = cublasGemmEx(bl, CUBLAS_OP_T, CUBLAS_OP_N,
                    N, 2048, K, &alpha, w, CUDA_R_16BF, K, x, CUDA_R_16BF, K,
                    &beta, o1, CUDA_R_32F, N, CUDA_R_32F, (cublasGemmAlgo_t)algo);
            cublasStatus_t s4 = cublasGemmEx(bl, CUBLAS_OP_T, CUBLAS_OP_N,
                    N, 4096, K, &alpha, w, CUDA_R_16BF, K, x, CUDA_R_16BF, K,
                    &beta, o2, CUDA_R_32F, N, CUDA_R_32F, (cublasGemmAlgo_t)algo);
            long fd2 = -2;
            if (s3 == CUBLAS_STATUS_SUCCESS && s4 == CUBLAS_STATUS_SUCCESS) {
                cudaDeviceSynchronize();
                cudaMemcpy(h1, o1, (uint64_t)2048 * N * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(h2, o2, (uint64_t)2048 * N * 4, cudaMemcpyDeviceToHost);
                fd2 = -1;
                for (long i = 0; i < (long)2048 * N; i++) if (h1[i] != h2[i]) { fd2 = i; break; }
            }
            printf("  algo %3d: 257v514 %s | 2048v4096 %s\n", algo,
                   fd < 0 ? "M-STABLE" : "differs",
                   fd2 == -2 ? "unsupported" : (fd2 < 0 ? "M-STABLE" : "differs"));
        }
        free(h1); free(h2);
        cudaFree(w); cudaFree(x); cudaFree(o1); cudaFree(o2);
    }
    cublasDestroy(bl);
    return 0;
}

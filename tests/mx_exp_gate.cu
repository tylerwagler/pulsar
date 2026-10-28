/* MX shared-exponent gate (upstream intake 2026-10-28) -- model-free, built with the engine's NVCCFLAGS, so it
 * measures what --use_fast_math makes of the device code.
 *
 * Every E4M3 activation producer derives its block's E8M0 exponent through pulsar_mx_shared_exp (pulsar_cuda_mx.cuh):
 * floor(log2(amax)) - 7, clamped to [-127, 127].  The CUTLASS grouped MoE's SwiGLU packer carried its own
 * floorf(log2f(amax)) until 2026-10-28; under fast math log2f is lg2.approx, one binade too high for the floats just
 * below each power of two, so those blocks got a 2x scale (one E4M3 bit fewer) -- the same failure the shared
 * helper's own comment records from 2026-10-05.  On every positive normal float:
 *   1. pulsar_mx_shared_exp(x) == the exact exponent (the float's IEEE exponent field, which needs no math library),
 *   2. the encode's inverse scale exp2f(-se) is the exact power of two for every se the helper can return
 *      (-127 .. 120: a finite amax is below 2^128).
 * It also prints how often the retired form errs (informational; on GB10, CUDA 13.3: 6896 floats high once clamped
 * as the packer clamped -- 7430 unclamped -- and 0 low) --
 * a low exponent would have been worse, saturating E4M3 past 448.
 * GPU, ~1 s. */
#include "cuda/pulsar_cuda_mx.cuh"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>

struct counts {
    unsigned long long helper_bad, naive_hi, naive_lo, inv_bad;
    uint32_t first_helper_bad;
};

/* the exact floor(log2(x)) - 7 of a positive normal float, clamped like the helper */
__device__ static int exact_se(float x) {
    int se = (int)((__float_as_uint(x) >> 23) & 0xffu) - 127 - 7;
    return se < -127 ? -127 : (se > 127 ? 127 : se);
}

__global__ static void sweep(counts *c, uint32_t base, uint32_t n) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float x = __uint_as_float(base + i);
        const int want = exact_se(x);
        const int se = pulsar_mx_shared_exp(x);
        if (se != want && atomicAdd(&c->helper_bad, 1ull) == 0ull) c->first_helper_bad = base + i;
        int naive = (int)floorf(log2f(x)) - 7;   /* the retired packer's form, clamped as it was */
        naive = naive < -127 ? -127 : (naive > 127 ? 127 : naive);
        if (naive > want) atomicAdd(&c->naive_hi, 1ull);
        if (naive < want) atomicAdd(&c->naive_lo, 1ull);
    }
}

__global__ static void inv_scales(counts *c) {
    const int se = (int)threadIdx.x - 127;   /* -127 .. 120: every exponent the helper returns */
    const float inv = exp2f((float)-se);
    if (__float_as_uint(inv) != (uint32_t)(127 - se) << 23) atomicAdd(&c->inv_bad, 1ull);
}

int main(void) {
    if (cudaSetDevice(0) != cudaSuccess) {
        fprintf(stderr, "MX-EXP FAIL: no GPU\n");
        return 1;
    }
    counts *c = nullptr;
    if (cudaMallocManaged(&c, sizeof(*c)) != cudaSuccess) {
        fprintf(stderr, "MX-EXP FAIL: cudaMallocManaged\n");
        return 1;
    }
    *c = counts{};
    const uint32_t lo = 0x00800000u, hi = 0x7f800000u;   /* every positive normal float */
    sweep<<<4096, 256>>>(c, lo, hi - lo);
    inv_scales<<<1, 248>>>(c);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "MX-EXP FAIL: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    int fail = 0;
    printf("mx-exp: %u positive normal floats\n", hi - lo);
    if (c->helper_bad) {
        printf("MX-EXP FAIL: pulsar_mx_shared_exp wrong on %llu floats (first 0x%08x)\n", c->helper_bad,
               c->first_helper_bad);
        fail = 1;
    } else {
        printf("mx-exp: pulsar_mx_shared_exp exact on all of them\n");
    }
    printf("mx-exp: (informational) the retired floorf(log2f()) form: %llu floats one binade high, %llu low\n",
           c->naive_hi, c->naive_lo);
    if (c->inv_bad) {
        printf("MX-EXP FAIL: exp2f(-se) is not the exact power of two for %llu exponents\n", c->inv_bad);
        fail = 1;
    } else {
        printf("mx-exp: exp2f(-se) exact for se in [-127, 120]\n");
    }
    printf(fail ? "MX-EXP FAIL\n" : "MX-EXP PASS\n");
    return fail;
}

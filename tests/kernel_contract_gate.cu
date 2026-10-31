/** @file
 *  Kernel contract gate (L335) -- model-free, links the PRODUCTION objects.
 *
 *  One case per L335 CUDA fix: each pins the contract the fix restored (an exact helper, a refusal, a bound, a
 *  NaN rule), so the fix cannot quietly come undone.  Built with the production NVCCFLAGS (--use_fast_math), which is
 *  the configuration the fixes were made for.
 *
 *    1. mx-shared-exp   pulsar_mx_shared_exp == the exact floor(log2(amax)) - 7 (double, on the host) on every binade
 *                       edge of the normal f32 range; prints how many the old fast-math floorf(log2f()) got wrong
 *                       (the expression swiglu_pack_e4m3_warp_kernel carried until L335).
 *
 *  Prints one line per case and KERNEL CONTRACT GATE PASS / FAIL. */
#include "pulsar_gpu.h"
#include "cuda/pulsar_cuda_mx.cuh"

#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

/// Failures so far (every case adds to it; main returns it as the verdict).
static int g_fail = 0;

/// Records one failed check and says which.
/// @param what the case and the check that failed
static void fail(const char *what) {
    fprintf(stderr, "KERNEL CONTRACT FAIL: %s\n", what);
    g_fail++;
}

/// True when a CUDA call returned cudaSuccess; otherwise records the failure by name.
/// @param e    the call's return code
/// @param what the call
/// @return e == cudaSuccess
static bool ok(cudaError_t e, const char *what) {
    if (e == cudaSuccess) return true;
    fprintf(stderr, "KERNEL CONTRACT FAIL: %s: %s\n", what, cudaGetErrorString(e));
    g_fail++;
    return false;
}

/* ---- 1. mx-shared-exp ------------------------------------------------------------------------------------------ */

/// The exponent the MXFP4 mid packer computed before L335 (kept HERE only, as the measured contrast).
/// @param mx the block amax
/// @return floorf(log2f(mx)) - 7, clamped to [-127, 127]; -127 for mx 0
__device__ static int old_fast_math_exp(float mx) {
    int se = -127;
    if (mx > 0.f) se = (int)floorf(log2f(mx)) - 7;
    if (se < -127) se = -127;
    if (se > 127) se = 127;
    return se;
}

/// One thread per value: the production exponent and the old one.
/// @param v    the amax values
/// @param n    how many
/// @param se   out: pulsar_mx_shared_exp(v[i])
/// @param old  out: old_fast_math_exp(v[i])
__global__ static void mx_exp_kernel(const float *v, int n, int *se, int *old) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    se[i] = pulsar_mx_shared_exp(v[i]);
    old[i] = old_fast_math_exp(v[i]);
}

/// The exact shared exponent, in double: floor(log2(a)) - 7 clamped to [-127, 127], -127 for 0.
/// @param a the amax
/// @return the exponent
static int exact_exp(float a) {
    if (!(a > 0.f)) return -127;
    int e = 0;
    frexp((double)a, &e);
    int se = (e - 1) - 7;
    if (se < -127) se = -127;
    if (se > 127) se = 127;
    return se;
}

/// Case 1: every binade edge 2^e (e = -126..127) -- the float just below, the power itself, the float just above.
static void case_mx_shared_exp(void) {
    std::vector<float> v;
    for (int e = -126; e <= 127; e++) {
        const float p = ldexpf(1.0f, e);
        v.push_back(nextafterf(p, 0.0f));
        v.push_back(p);
        v.push_back(nextafterf(p, INFINITY));
    }
    v.push_back(0.0f);
    const int n = (int)v.size();
    float *dv = nullptr;
    int *dse = nullptr, *dold = nullptr;
    if (!ok(cudaMalloc(&dv, n * sizeof(float)), "mx-shared-exp malloc") ||
        !ok(cudaMalloc(&dse, n * sizeof(int)), "mx-shared-exp malloc") ||
        !ok(cudaMalloc(&dold, n * sizeof(int)), "mx-shared-exp malloc")) return;
    std::vector<int> se(n), old(n);
    if (ok(cudaMemcpy(dv, v.data(), n * sizeof(float), cudaMemcpyHostToDevice), "mx-shared-exp upload")) {
        mx_exp_kernel<<<(n + 127) / 128, 128>>>(dv, n, dse, dold);
        if (ok(cudaDeviceSynchronize(), "mx-shared-exp kernel") &&
            ok(cudaMemcpy(se.data(), dse, n * sizeof(int), cudaMemcpyDeviceToHost), "mx-shared-exp download") &&
            ok(cudaMemcpy(old.data(), dold, n * sizeof(int), cudaMemcpyDeviceToHost), "mx-shared-exp download")) {
            int bad = 0, old_bad = 0;
            for (int i = 0; i < n; i++) {
                const int want = exact_exp(v[i]);
                if (se[i] != want) {
                    if (bad < 4) fprintf(stderr, "  mx-shared-exp: amax %a -> %d, exact %d\n", v[i], se[i], want);
                    bad++;
                }
                if (old[i] != want) old_bad++;
            }
            printf("  1 mx-shared-exp: %d binade-edge values, production %d off, old fast-math floorf(log2f) %d off\n",
                   n, bad, old_bad);
            if (bad) fail("mx-shared-exp: pulsar_mx_shared_exp differs from the exact exponent");
        }
    }
    cudaFree(dv);
    cudaFree(dse);
    cudaFree(dold);
}

/// Runs every case on device 0.
/// @return 0 on PASS, 1 on FAIL
int main(void) {
    if (cudaSetDevice(0) != cudaSuccess) {
        fprintf(stderr, "KERNEL CONTRACT FAIL: no GPU\n");
        return 1;
    }
    case_mx_shared_exp();
    puts(g_fail ? "KERNEL CONTRACT GATE FAIL" : "KERNEL CONTRACT GATE PASS");
    return g_fail ? 1 : 0;
}

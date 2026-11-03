/* mma_k8_split_gate.cu -- L338: the f16 m16n8k16 MMA's sm_75 body (two m16n8k8, pulsar_mma_f16_16816_k8) graded
 * against the exact product and against the k16 instruction, on an sm_80+ card that has both.
 *
 *   mma_k8_split_gate [N_TRIALS]
 *
 * THE FINDING THIS GATE RECORDS (2026-11-01, GB10 sm_121): the split is NOT bit-identical to the k16 instruction --
 * about a third of the accumulators differ, by an ulp or a few, more where a large partial sum cancels -- because each
 * k8 half rounds its partial sum to f32 once more (exllamav3's sm_75 port and plan 123 called the split bit-equivalent;
 * it is not).  So the sm_75 body is a numerics change, graded here: both arithmetics must be accurate f32 accumulations
 * of the same terms.  For every lane and accumulator the exact value e = d0 + sum_k a_k b_k is computed in double from
 * the trial's whole A (16x16) / B (16x8), reassembled from the fragments, with S = |d0| + sum_k |a_k b_k|; each result
 * x must satisfy |x - e| <= TOL * 2^-24 * S.  The tensor core does not accumulate in IEEE round-to-nearest, so TOL is
 * calibrated, not ideal f32: measured on GB10 (65536 trials, 8.4 M accumulators, 2026-11-01) the k16 instruction's own
 * worst is 4.80 units and the split's 6.24 (35.1 % of the accumulators differ in their bits); TOL 8 (= 2^-21 S) holds
 * both with margin.  The worst errors and the bit-mismatch rate are printed as the record.  Exit 0 = both within the bound on every lane of every trial.  Model-free; refuses below sm_80
 * (the k16 instruction is the comparison).
 *
 * Fragment layout (PTX ISA, mma.m16n8k16 .f16 row.col; lane = 4 g + c): a[0] = A[g][2c, 2c+1], a[1] = A[g+8][2c, 2c+1],
 * a[2] = A[g][2c+8, 2c+9], a[3] = A[g+8][2c+8, 2c+9]; b0 = B[2c, 2c+1][g], b1 = B[2c+8, 2c+9][g];
 * d[0] = D[g][2c], d[1] = D[g][2c+1], d[2] = D[g+8][2c], d[3] = D[g+8][2c+1]. */
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../src/cuda/pulsar_cuda_arch.cuh"

/// The bound, in units of 2^-24 * S (S = |d0| + sum |a b|).
static const double TOL = 8.0;

/* one trial per warp: in = [A 4 regs | B 2 regs | d 4 floats] per lane (10 words), out = [d_k16 4 | d_k8 4] per lane */
__global__ void split_kernel(const uint32_t *in, uint32_t *out, int n_trials) {
    const int lane = threadIdx.x & 31, trial = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    if (trial >= n_trials) return;
    const uint32_t *p = in + ((size_t)trial * 32 + lane) * 10;
    const uint32_t a[4] = {p[0], p[1], p[2], p[3]};
    float d16[4], d8[4];
    for (int i = 0; i < 4; ++i) d16[i] = d8[i] = __uint_as_float(p[6 + i]);
    pulsar_mma_f16_16816(d16, a, p[4], p[5]);       /* sm_80+: the k16 instruction */
    pulsar_mma_f16_16816_k8(d8, a, p[4], p[5]);     /* the sm_75 body */
    uint32_t *o = out + ((size_t)trial * 32 + lane) * 8;
    for (int i = 0; i < 4; ++i) {
        o[i] = __float_as_uint(d16[i]);
        o[4 + i] = __float_as_uint(d8[i]);
    }
}

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t rnd() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}
static float urand() { return (float)((rnd() >> 11) * (1.0 / 9007199254740992.0)); }

/* an fp16 operand: mostly normal values of mixed scale, some subnormals, signed zeros and exact powers of two */
static uint16_t f16_operand() {
    const uint32_t k = (uint32_t)(rnd() % 100);
    float v;
    if (k < 70) v = (urand() * 2.0f - 1.0f) * powf(2.0f, (float)((int)(rnd() % 12) - 6));
    else if (k < 80) v = (urand() * 2.0f - 1.0f) * 6.0e-5f;          /* fp16 subnormals */
    else if (k < 88) v = (rnd() & 1) ? 0.0f : -0.0f;
    else v = ((rnd() & 1) ? 1.0f : -1.0f) * powf(2.0f, (float)((int)(rnd() % 16) - 8));
    return __half_as_ushort(__float2half_rn(v));
}

static double h2d(uint16_t h) { return (double)__half2float(__ushort_as_half(h)); }

int main(int argc, char **argv) {
    const int n_trials = argc > 1 ? atoi(argv[1]) : 1 << 16;
    int dev = 0, ma = 0, mi = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&ma, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&mi, cudaDevAttrComputeCapabilityMinor, dev);
    if (ma < 8) {
        fprintf(stderr, "mma-k8-split gate: sm_%d%d has no m16n8k16 to grade the split against -- refusing (run it on an "
                        "sm_80+ card)\n", ma, mi);
        return 2;
    }
    const size_t n_lanes = (size_t)n_trials * 32;
    std::vector<uint32_t> h_in(n_lanes * 10), h_out(n_lanes * 8);
    for (size_t i = 0; i < n_lanes; ++i) {
        uint32_t *p = &h_in[i * 10];
        for (int r = 0; r < 6; ++r) p[r] = (uint32_t)f16_operand() | (uint32_t)f16_operand() << 16;
        for (int r = 0; r < 4; ++r) {
            const float acc = (urand() * 2.0f - 1.0f) * powf(2.0f, (float)((int)(rnd() % 20) - 10));
            memcpy(&p[6 + r], &acc, 4);
        }
    }
    uint32_t *d_in = nullptr, *d_out = nullptr;
    if (cudaMalloc(&d_in, h_in.size() * 4) || cudaMalloc(&d_out, h_out.size() * 4)) {
        fprintf(stderr, "mma-k8-split gate: cudaMalloc failed\n");
        return 1;
    }
    cudaMemcpy(d_in, h_in.data(), h_in.size() * 4, cudaMemcpyHostToDevice);
    split_kernel<<<(n_trials + 7) / 8, 256>>>(d_in, d_out, n_trials);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "mma-k8-split gate: the kernel failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    cudaMemcpy(h_out.data(), d_out, h_out.size() * 4, cudaMemcpyDeviceToHost);

    uint64_t differ = 0, over16 = 0, over8 = 0;
    double worst16 = 0.0, worst8 = 0.0;
    for (int t = 0; t < n_trials; ++t) {
        double A[16][16], B[16][8];   /* reassembled from the 32 lanes' fragments */
        for (int lane = 0; lane < 32; ++lane) {
            const uint32_t *p = &h_in[((size_t)t * 32 + lane) * 10];
            const int g = lane >> 2, c = lane & 3;
            const int ar[4] = {g, g + 8, g, g + 8}, ak[4] = {2 * c, 2 * c, 2 * c + 8, 2 * c + 8};
            for (int r = 0; r < 4; ++r) {
                A[ar[r]][ak[r]] = h2d((uint16_t)(p[r] & 0xffffu));
                A[ar[r]][ak[r] + 1] = h2d((uint16_t)(p[r] >> 16));
            }
            B[2 * c][g] = h2d((uint16_t)(p[4] & 0xffffu));
            B[2 * c + 1][g] = h2d((uint16_t)(p[4] >> 16));
            B[2 * c + 8][g] = h2d((uint16_t)(p[5] & 0xffffu));
            B[2 * c + 9][g] = h2d((uint16_t)(p[5] >> 16));
        }
        for (int lane = 0; lane < 32; ++lane) {
            const uint32_t *p = &h_in[((size_t)t * 32 + lane) * 10], *o = &h_out[((size_t)t * 32 + lane) * 8];
            const int g = lane >> 2, c = lane & 3;
            const int dr[4] = {g, g, g + 8, g + 8}, dc[4] = {2 * c, 2 * c + 1, 2 * c, 2 * c + 1};
            for (int r = 0; r < 4; ++r) {
                float d0, x16, x8;
                memcpy(&d0, &p[6 + r], 4);
                memcpy(&x16, &o[r], 4);
                memcpy(&x8, &o[4 + r], 4);
                double e = d0, S = fabs((double)d0);
                for (int k = 0; k < 16; ++k) {
                    const double pr = A[dr[r]][k] * B[k][dc[r]];
                    e += pr;
                    S += fabs(pr);
                }
                const double unit = ldexp(S, -24);
                const double r16 = unit > 0 ? fabs((double)x16 - e) / unit : 0.0, r8 = unit > 0 ? fabs((double)x8 - e) / unit : 0.0;
                if (r16 > worst16) worst16 = r16;
                if (r8 > worst8) worst8 = r8;
                over16 += r16 > TOL;
                if (r8 > TOL && over8++ < 8)
                    fprintf(stderr, "  trial %d lane %d acc %d: exact %.12g, k16 %.9g (%.2f), k8 split %.9g (%.2f) in units "
                                    "of 2^-24 S\n", t, lane, r, e, (double)x16, r16, (double)x8, r8);
                differ += o[r] != o[4 + r];
            }
        }
    }
    const double total = (double)n_lanes * 4;
    printf("mma-k8-split gate (sm_%d%d), %d trials x 32 lanes x 4 accumulators:\n"
           "  bits differ from the k16 instruction: %llu (%.1f%%) -- a numerics change, not a per-target body\n"
           "  worst error vs exact, in units of 2^-24 * (|d0| + sum |a b|): k16 %.3f, k8 split %.3f (bound %.1f)\n"
           "  over the bound: k16 %llu, k8 split %llu -> %s\n",
           ma, mi, n_trials, (unsigned long long)differ, 100.0 * (double)differ / total, worst16, worst8, TOL,
           (unsigned long long)over16, (unsigned long long)over8,
           over16 || over8 ? "FAIL" : "PASS (both accurate f32 accumulations of the same terms)");
    cudaFree(d_in);
    cudaFree(d_out);
    return over16 || over8 ? 1 : 0;
}

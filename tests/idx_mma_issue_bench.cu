/* Is the block-scaled MMA itself the ceiling?
 *
 * Every feeding-side lever on the indexer scorer has now been measured and
 * ruled out: operand traffic, bank conflicts, staging, register churn, fetch
 * latency, and occupancy (50% -> 83% moved the runtime 0.3%).  What is left is
 * the instruction.  This times it with NO memory traffic at all -- operands
 * live in registers, the loop is unrolled, nothing is loaded or stored inside
 * the timed region -- so what comes out is pure issue rate.
 *
 * Compares the two candidates:
 *   kind::mxf8f6f4  m16n8k32  e4m3 x e2m1   (what the scorer uses; 4096 MAC/inst)
 *   kind::mxf4nvf4  m16n8k64  e2m1 x e2m1   (8192 MAC/inst, packed nibbles)
 *
 * If they issue at similar rates, the scorer is near the instruction's ceiling
 * and the remaining gap to Entrpi is not in this kernel.  If mxf4nvf4 is much
 * faster, the fidelity fork settled in b2c39ad was priced on a false premise
 * (it assumed equal speed) and has to be re-decided.
 *
 * build+run (single arch target: block-scaled asm cannot go through the
 * compute_120 PTX pass -- see tests/idx_mxfp4_probe.cu):
 *   nvcc -O3 -gencode arch=compute_121a,code=sm_121a -o /tmp/mmabench \
 *        tests/idx_mma_issue_bench.cu
 *   /tmp/mmabench
 */
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define ITERS 2048

__global__ static void bench_mxf8f6f4(float *sink, int iters) {
    uint32_t a0 = 0x38383838u, a1 = a0, a2 = a0, a3 = a0;
    uint32_t b0 = 0x08080808u, b1 = b0;      /* e2m1 nibble 2 at bits [5:2] */
    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    const uint32_t sfa = 128u, sfb = 126u;
    const uint16_t bid = 0, tid = 0;
    for (int i = 0; i < iters; i++) {
        #pragma unroll 8
        for (int j = 0; j < 8; j++) {
            asm volatile(
                "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "r"(sfa), "h"(bid), "h"(tid), "r"(sfb), "h"(bid), "h"(tid));
        }
    }
    /* MUST be observable: with an unreachable store the accumulator chain is
     * dead and the timings come out physically impossible (19,522 TMAC/s). */
    if (threadIdx.x == 0) sink[blockIdx.x] = d0 + d1 + d2 + d3;
}

/* The attention question (docs/engine-perf-map.md): attention is 22-25% of
 * prefill with pipe_tensor at 0%, and moving it onto the tensor cores means
 * choosing an operand format.  pulsar_cuda_attn_f16.cu's header measures what
 * each format COSTS; these two measure what fp16 and bf16 BUY, against the
 * block-scaled rates above and against the FP32 FMA pipe it runs on today.
 * m16n8k16 with f32 accumulate = 2048 MAC/inst. */
__global__ static void bench_f16(float *sink, int iters) {
    uint32_t a0 = 0x3C003C00u, a1 = a0, a2 = a0, a3 = a0;   /* 1.0h x2 */
    uint32_t b0 = 0x3C003C00u, b1 = b0;
    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    for (int i = 0; i < iters; i++) {
        #pragma unroll 8
        for (int j = 0; j < 8; j++) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    if (threadIdx.x == 0) sink[blockIdx.x] = d0 + d1 + d2 + d3;
}

__global__ static void bench_bf16(float *sink, int iters) {
    uint32_t a0 = 0x3F803F80u, a1 = a0, a2 = a0, a3 = a0;   /* 1.0bf16 x2 */
    uint32_t b0 = 0x3F803F80u, b1 = b0;
    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    for (int i = 0; i < iters; i++) {
        #pragma unroll 8
        for (int j = 0; j < 8; j++) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    if (threadIdx.x == 0) sink[blockIdx.x] = d0 + d1 + d2 + d3;
}

/* What attention runs on TODAY: the FP32 FMA pipe.  Same harness so the
 * comparison is apples to apples -- a dependent FFMA chain per thread, no
 * memory in the loop.  2 flops per FFMA = 1 MAC. */
__global__ static void bench_ffma(float *sink, int iters) {
    float a = 1.000001f, d0 = 1.f, d1 = 1.f, d2 = 1.f, d3 = 1.f;
    for (int i = 0; i < iters; i++) {
        #pragma unroll 8
        for (int j = 0; j < 8; j++) {
            d0 = fmaf(d0, a, 1.f); d1 = fmaf(d1, a, 1.f);
            d2 = fmaf(d2, a, 1.f); d3 = fmaf(d3, a, 1.f);
        }
    }
    if (threadIdx.x == 0) sink[blockIdx.x] = d0 + d1 + d2 + d3;
}

__global__ static void bench_mxf4nvf4(float *sink, int iters) {
    uint32_t a0 = 0x22222222u, a1 = a0, a2 = a0, a3 = a0;   /* packed e2m1 nibbles */
    uint32_t b0 = 0x22222222u, b1 = b0;
    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    const uint32_t sfa = 128u, sfb = 128u;
    const uint16_t bid = 0, tid = 0;
    for (int i = 0; i < iters; i++) {
        #pragma unroll 8
        for (int j = 0; j < 8; j++) {
            asm volatile(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "r"(sfa), "h"(bid), "h"(tid), "r"(sfb), "h"(bid), "h"(tid));
        }
    }
    if (threadIdx.x == 0) sink[blockIdx.x] = d0 + d1 + d2 + d3;
}

template <typename K>
static double run(K kern, int blocks, int threads, const char *tag, double mac_per_inst) {
    float *sink = nullptr;
    cudaMalloc(&sink, sizeof(float) * 4096);
    kern<<<blocks, threads>>>(sink, 8);            /* warm */
    if (cudaDeviceSynchronize() != cudaSuccess) {
        printf("  %-12s UNSUPPORTED: %s\n", tag, cudaGetErrorString(cudaGetLastError()));
        cudaFree(sink); return 0.0;
    }
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    kern<<<blocks, threads>>>(sink, ITERS);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        printf("  %-12s TIMED LAUNCH FAILED: %s\n", tag, cudaGetErrorString(cudaGetLastError()));
        cudaFree(sink); return 0.0;
    }
    float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1);
    const double insts = (double)blocks * (threads / 32) * ITERS * 8.0;
    const double tmac = insts * mac_per_inst / (ms * 1e-3) / 1e12;
    printf("  %-12s %8.3f ms   %8.2f G inst/s   %8.1f TMAC/s\n",
           tag, ms, insts / (ms * 1e-3) / 1e9, tmac);
    cudaFree(sink);
    return tmac;
}

int main(void) {
    int sm = 0; cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);
    const int blocks = sm * 4, threads = 256;
    printf("MMA issue-rate bench: %d SMs, %d blocks x %d threads, no memory in the loop\n\n",
           sm, blocks, threads);
    const double t8 = run(bench_mxf8f6f4, blocks, threads, "mxf8f6f4", 4096.0);
    const double t4 = run(bench_mxf4nvf4, blocks, threads, "mxf4nvf4", 8192.0);
    if (t8 > 0 && t4 > 0)
        printf("\n  mxf4nvf4 / mxf8f6f4 = %.2fx on MACs\n", t4 / t8);

    /* The attention format question: what each candidate buys. */
    printf("\n  attention operand candidates (see pulsar_cuda_attn_f16.cu\n"
           "  for what each one COSTS):\n");
    const double tf16 = run(bench_f16,  blocks, threads, "f16 m16n8k16",  2048.0);
    const double tbf  = run(bench_bf16, blocks, threads, "bf16 m16n8k16", 2048.0);
    /* run() counts 8 units per warp per outer iteration; each unit here is
     * 4 FFMA per thread x 32 lanes = 128 MACs (an MMA unit is one instruction
     * for the whole warp, an FFMA unit is one per lane). */
    const double tfm  = run(bench_ffma, blocks, threads, "f32 FMA pipe",  128.0);
    if (tfm > 0) {
        if (tf16 > 0) printf("\n  f16  vs the FP32 FMA pipe attention uses today: %.1fx\n", tf16 / tfm);
        if (tbf  > 0) printf("  bf16 vs the FP32 FMA pipe attention uses today: %.1fx\n", tbf / tfm);
        if (t8   > 0) printf("  fp8 (block-scaled) vs that same pipe:           %.1fx\n", t8 / tfm);
    }
    /* 6.5 TMAC/s was total MACs over a runtime that was 82% Q pack.  Measured
     * against the n_comp slope alone the GEMM does ~33 TMAC/s, so the real
     * comparison is 33 against the 125 below: a 3.8x headroom, not a 19x one. */
    printf("\n  scorer GEMM (n_comp slope only) achieves ~33 TMAC/s\n");
    printf("BENCH_DONE\n");
    return 0;
}

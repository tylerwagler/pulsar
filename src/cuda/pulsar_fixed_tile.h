/**
 * @file pulsar_fixed_tile.h
 * @brief L151-C: fixed-tile tensor-core GEMMs for the M-neutral (armed) dense
 *        step -- one tile configuration regardless of M, so a row's bytes never
 *        depend on how many rows share the call, and the weight streams once
 *        for all rows (cuBLASLt-flat where the nt kernels grow linearly in M).
 *
 * Plain-pointer interface between pulsar_cuda_matmul.cu (the dispatch, which
 * owns weight resolution, the activation slots and the output type) and
 * pulsar_fixed_tile.cu (the CUTLASS unit, compiled like the MXFP4 expert GEMM:
 * nvcc, C++17, CUTLASS include path).  Nothing here is reachable unless the
 * step is armed (pulsar_gpu_matmul_set_batch_mneutral > 0).
 *
 * Priced by the stage-0 probe (tests/fixed_tile_gemm_probe.cpp,
 * pulsar-notes rows/L151.md, 2026-09-03): byte-neutral across M on every
 * shape, within 1.15x of cuBLASLt at 9 rows, bit-identical to cuBLASLt for
 * M >= 5, no weight repack (the LT scale slab IS the CUTLASS scale layout).
 */
#ifndef PULSAR_FIXED_TILE_H
#define PULSAR_FIXED_TILE_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/** Does the MXFP8 fixed tile take this dense shape?  Stage 1 scope: the two
 * 34 MB projections (attn_q_b, attn_output_b) -- weight bytes >= 16 MiB, K a
 * multiple of 128, N a multiple of 64.  The smaller shapes stay on the nt
 * kernels: at N <= 4096 a 128-wide tile is 8-32 CTAs and slower than nt at
 * M = 1 (the stage-0 "CTA floor"); they wait for a narrower tile or a fixed
 * split-K.  Shape-derived on purpose: no per-tensor list to drift. */
int pulsar_ft_mxfp8_shape_ok(uint64_t in_dim, uint64_t out_dim);

/** MXFP8 x MXFP8 -> f32 (or f16 when out_f16) with a fixed 128 x TN x 128
 * block-scaled tile, TN = 128 (N >= 16384) or 64.
 * @param out      destination rows [M][N], f32 or f16 per out_f16
 * @param xq       E4M3 activations [M][K] row-major (the A8 slot's xq)
 * @param sx       their UE8M0 block scales in the VEC32 swizzle
 *                 (pulsar_mx_sfoff: rows padded to 128, blocks to 4) -- the same
 *                 function as CUTLASS's SFA layout, asserted at first use per K
 * @param wdata    weight E4M3 [N][K] K-major (the MXFP8_LT data slab)
 * @param wscale   weight UE8M0 in the same swizzle (the LT scale slab)
 * @param M,N,K    rows (1..16), out_dim, in_dim
 * @param ws       caller workspace (may be NULL when ws_bytes is 0); the kernel
 *                 refuses rather than allocate if it needs more
 * @return 1 on success, 0 on refusal or launch failure (announced once). */
int pulsar_ft_mxfp8(void *out, int out_f16,
                    const uint8_t *xq, const uint8_t *sx,
                    const uint8_t *wdata, const uint8_t *wscale,
                    int M, int N, int K, void *ws, size_t ws_bytes);

/** Does the bf16 fixed tile take this shape?  Stage 1 scope: the output head
 * only -- weight bytes >= 512 MiB, K a multiple of 64, N a multiple of 128.
 * The router (2 MB, N = 256) stays on nt: 2-4 CTAs, not worth a kernel. */
int pulsar_ft_bf16_shape_ok(uint64_t in_dim, uint64_t out_dim);

/** bf16 x bf16 -> f32 with a fixed 64 x 128 x 64 mma.sync tile (4 stages,
 * no split-K).  xb = the call's bf16 activations [M][K]; w = bf16 [N][K]. */
int pulsar_ft_bf16(float *out, const uint16_t *xb, const uint16_t *w,
                   int M, int N, int K, void *ws, size_t ws_bytes);

#ifdef __cplusplus
}
#endif
#endif

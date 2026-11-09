/* L151-C stage 0: C interface between the CUTLASS probe unit
 * (fixed_tile_gemm_kernels.cu, nvcc/C++17) and the engine-client driver
 * (fixed_tile_gemm_probe.cpp, C++23).  Pricing only; not engine API. */
#ifndef FIXED_TILE_GEMM_PROBE_H
#define FIXED_TILE_GEMM_PROBE_H
#include <stdint.h>
#include "pulsar_gpu.h"
#ifdef __cplusplus
extern "C" {
#endif
struct ft_ctx;
/** Upload one MXFP8_LT weight (its two slabs, host pointers into the mmap),
 * re-lay the scale into CUTLASS's SFB layout, size workspaces for M <= 16.
 * sfb_mismatch_out receives how many (n, kb) scale bytes sit at a different
 * linear offset in the CUTLASS layout than in the LT slab (0 = same swizzle). */
struct ft_ctx *ft_prepare(const uint8_t *host_lt_data, const uint8_t *host_lt_scale, int N, int K,
                          int m_max, unsigned long long *sfb_mismatch_out);
/** Quantise x[M,K] f32 (device) to the engine's E4M3 bytes and run the
 * fixed-tile GEMM with tile N = tn (64 or 128) into D[M,N] f32 (device).
 * Returns 0 on success, 1 = can_implement refused, 2/3 = init/run failed. */
int ft_run(struct ft_ctx *c, int tn, const pulsar_gpu_tensor *x, int M, pulsar_gpu_tensor *D);
int ft_sync(void);
void ft_release(struct ft_ctx *c);

/** Stage 0.4, the bf16 family: upload a bf16 [N][K] weight (host pointer
 * into the mmap) and run a fixed 64 x tn x 32 mma.sync tile GEMM on the
 * engine's bf16-rounded activations.  Same return codes as ft_run. */
struct fb_ctx;
struct fb_ctx *fb_prepare(const uint8_t *host_w_bf16, int N, int K, int m_max);
int fb_run(struct fb_ctx *c, int tn, const pulsar_gpu_tensor *x, int M, pulsar_gpu_tensor *D);
/* L183: emit the engine-format bf16 activation plane for x's first M rows into xb
 * (the slot from pulsar_gpu_bf16_act_slot), so the engine's bf16 arm has a producer. */
int fb_emit_plane(const pulsar_gpu_tensor *x, int M, int K, void *xb);
/* L183: cuBLASLt bf16 with the reduction scheme pinned to NONE -- the candidate
 * neutral arm for the plain-weight family; heuristic per M like the engine's
 * shape cache.  Same return codes as ft_run. */
struct lt_ctx *lt_prepare(const uint16_t *host_w_bf16, int N, int K, int m_max);
int lt_emit_plane(struct lt_ctx *c, const pulsar_gpu_tensor *x, int M);   /* once per shape, before lt_run */
int lt_run(struct lt_ctx *c, const pulsar_gpu_tensor *x, int M, pulsar_gpu_tensor *D);
void lt_release(struct lt_ctx *c);
void fb_release(struct fb_ctx *c);
#ifdef __cplusplus
}
#endif
#endif

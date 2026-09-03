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
                          unsigned long long *sfb_mismatch_out);
/** Quantise x[M,K] f32 (device) to the engine's E4M3 bytes and run the
 * fixed-tile GEMM with tile N = tn (64 or 128) into D[M,N] f32 (device).
 * Returns 0 on success, 1 = can_implement refused, 2/3 = init/run failed. */
int ft_run(struct ft_ctx *c, int variant, const pulsar_gpu_tensor *x, int M, pulsar_gpu_tensor *D);
int ft_nvariants(void);
const char *ft_variant_name(int v);
int ft_sync(void);
void ft_release(struct ft_ctx *c);

/** Stage 0.4, the bf16 family: upload a bf16 [N][K] weight (host pointer
 * into the mmap) and run a fixed 64 x tn x 32 mma.sync tile GEMM on the
 * engine's bf16-rounded activations.  Same return codes as ft_run. */
struct fb_ctx;
struct fb_ctx *fb_prepare(const uint8_t *host_w_bf16, int N, int K);
int fb_run(struct fb_ctx *c, int variant, const pulsar_gpu_tensor *x, int M, pulsar_gpu_tensor *D);
int fb_nvariants(void);
const char *fb_variant_name(int v);
void fb_release(struct fb_ctx *c);
#ifdef __cplusplus
}
#endif
#endif

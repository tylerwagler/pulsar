/* L151-C stage 0 / L169: C interface between the CUTLASS probe unit
 * (fixed_tile_gemm_kernels.cu, nvcc/C++17) and the engine-client driver
 * (fixed_tile_gemm_probe.cpp, C++23).  Pricing only; not engine API.
 *
 * Three objects per format: a WEIGHT (one layer's slab on device), an ACT
 * (the packed activation rows + CUTLASS workspace, shared by every weight of
 * the shape) and a PLAN (one (weight, variant, M, D) launch with its CUTLASS
 * Params -- TMA descriptors, scheduler decomposition -- built once, so the
 * timed loop pays only what the engine's producer/consumer split would pay
 * per call: a barrier-workspace reset and the kernel launch). */
#ifndef FIXED_TILE_GEMM_PROBE_H
#define FIXED_TILE_GEMM_PROBE_H
#include <stdint.h>
#include "pulsar_gpu.h"
#ifdef __cplusplus
extern "C" {
#endif
/** Multiprocessor count and L2 size of device 0 (cudaDeviceGetAttribute). 0 on success. */
int ft_device_info(int *sm_count, unsigned long long *l2_bytes);

/** How a variant's tile scheduler decomposed the problem, read back from the
 * CUTLASS Params after initialize(): the EFFECTIVE split count (CUTLASS clamps
 * splits to the K-tile count and the SM count), the stream-K unit/tile counts
 * (0 for split-K and for the plain tile), the K-slices that carry one extra K
 * tile (big_units; fixed slice boundaries), and the launched CTA count. */
typedef struct { int splits; unsigned sk_units, sk_tiles, big_units, ctas; } ft_decomp;

/* ---- MXFP8: 128x128x128 block-scaled tile; plain / split-K / stream-K ---- */
struct ft_weight;
struct ft_act;
struct ft_plan;
int ft_nvariants(void);
const char *ft_variant_name(int v);
/** Upload one MXFP8_LT weight (its two slabs, host pointers into the mmap) and
 * re-lay the scale into CUTLASS's SFB layout.  sfb_mismatch_out receives how
 * many (n, kb) scale bytes sit at a different linear offset in the CUTLASS
 * layout than in the LT slab (0 = same swizzle). */
struct ft_weight *ft_prepare(const uint8_t *host_lt_data, const uint8_t *host_lt_scale, int N, int K,
                             unsigned long long *sfb_mismatch_out);
/** Activation rows (<= 16) and one workspace sized for every variant and M. */
struct ft_act *ft_act_prepare(int N, int K);
/** Quantise x[M,K] f32 (device) to the engine's E4M3 bytes + E8M0 scales in
 * the CUTLASS SFA layout -- the producer's job in the engine, so it sits
 * OUTSIDE the timed loop.  0 on success. */
int ft_pack(struct ft_act *a, const pulsar_gpu_tensor *x, int M);
/** Build the launch for (weight, variant, M) writing D[M,N] f32 (device).
 * 0 and *out on success; 1 = can_implement refused, 2 = initialize failed,
 * 4 = bad handle, 5 = undersized D, 6 = unknown variant. */
int ft_plan_make(struct ft_weight *w, struct ft_act *a, int variant, int M, pulsar_gpu_tensor *D,
                 struct ft_plan **out);
/** Reset the scheduler's barrier workspace and launch.  0 on success, 3 = run failed. */
int ft_plan_run(struct ft_plan *p);
void ft_plan_decomp(const struct ft_plan *p, ft_decomp *d);
void ft_plan_release(struct ft_plan *p);
int ft_sync(void);
void ft_release(struct ft_weight *w);
void ft_act_release(struct ft_act *a);

/* ---- bf16 (router, output head): fixed mma.sync tiles, same object model ---- */
struct fb_weight;
struct fb_act;
struct fb_plan;
int fb_nvariants(void);
const char *fb_variant_name(int v);
struct fb_weight *fb_prepare(const uint8_t *host_w_bf16, int N, int K);
struct fb_act *fb_act_prepare(int N, int K);
/** Round x[M,K] f32 -> bf16 with the engine's RNE; outside the timed loop. */
int fb_pack(struct fb_act *a, const pulsar_gpu_tensor *x, int M);
int fb_plan_make(struct fb_weight *w, struct fb_act *a, int variant, int M, pulsar_gpu_tensor *D,
                 struct fb_plan **out);
int fb_plan_run(struct fb_plan *p);
void fb_plan_release(struct fb_plan *p);
void fb_release(struct fb_weight *w);
void fb_act_release(struct fb_act *a);
#ifdef __cplusplus
}
#endif
#endif

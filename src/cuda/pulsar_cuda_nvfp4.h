// SPDX-License-Identifier: MIT
// L298: NVFP4 W4A16 -- an NVFP4 weight (ModelOpt W4A16_NVFP4: e2m1 values, an E4M3 scale per 16 inputs, one f32
// global scale) against raw bf16 rows.  A FORMAT, not a family's: any family's dense Linear stored `nvfp4` takes it
// (weight_format.cpp PULSAR_DENSE_ARM_NVFP4_ROWS).
//
// The stored tensor (layout `nvfp4`, dims_ne [in, out]) is one slice, three planes back to back:
//   values  [out][in / 2]  u8: two e2m1 codes a byte, input 2i in the LOW nibble, 2i + 1 in the high one (ModelOpt's
//                              packing; vLLM break_fp4_bytes)
//   scales  the 128x4-tiled E4M3 plane (the NVFP4 MMA layout cuBLASLt, CUTLASS, FlashInfer and b12x read): row r's
//           scale of inputs [16 kb, 16 kb + 16) at byte  ((r / 128) * (in / 64) + kb / 4) * 512 + (r % 32) * 16
//           + ((r % 128) / 32) * 4 + kb % 4, out padded to 128 rows (zero scales)
//   global  one f32 (weight_scale_2), then 28 zero bytes (a 32-byte tail: the container starts every packed entry
//           32-byte aligned, and values and scales are multiples of 32)
// The weight is w[r][k] = e2m1 * e4m3 * global.  pulsar_nvfp4_layout is the one byte model (the loader's and the
// kernel's).
//
// The arithmetic (pulsar_cuda_nvfp4.cu): e2m1 x e4m3 is exact in bf16 (at most 2 x 4 significant bits), so the kernel
// dequantizes each value to that bf16 IN REGISTERS and runs the BF16 rows kernel's order on it -- the same 64-input
// steps, k permutation, m16n8k16 fragments and fresh-fragment-added-in-k-order accumulation
// (pulsar_cuda_bf16_rows.cu) -- then multiplies the f32 sum by the global scale.  With K split S ways (a function of
// the shape alone, pulsar_nvfp4_ksplit) each slice is that sum over its inputs and the slices add in slice order.
// So y is BIT-IDENTICAL to the BF16 rows kernel run on the dequantized (bf16) weight -- per K slice, slices summed in
// order -- times the global scale: the BF16 bring-up path's bytes when S = 1.  A row's bytes do not depend on the
// other rows of the call (M-independent at every width).
#pragma once

#include <cuda_runtime.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/** Bytes of the planes of an [out][in] nvfp4 slice (in % 64 == 0, out > 0); false = not an nvfp4 shape. */
static inline bool pulsar_nvfp4_layout(uint64_t in, uint64_t out, uint64_t *values, uint64_t *scales,
                                       uint64_t *stride) {
    if (in == 0 || out == 0 || in % 64 != 0) return false;
    *values = out * in / 2;
    *scales = (out + 127) / 128 * 128 * (in / 16);
    *stride = *values + *scales + 32;
    return true;
}

/** The K split of an [out][in] nvfp4 Linear: the smallest S in 1, 2, 4, 8 giving >= 96 CTAs of 128 / S rows each (a
 *  GB10 has 48 SMs) with in % (64 S) == 0.  A function of the shape alone: it fixes the summation order. */
int pulsar_nvfp4_ksplit(int in, int out);

/** y [rows][out] f32 = W x (and, when y_bf16 is non-NULL, the same values rounded to nearest-even bf16 in
 *  y_bf16 [rows][out]) for an nvfp4 slice `w` (pulsar_nvfp4_layout) and bf16 rows x [rows][in].  w, x 16-byte
 *  aligned.  Every row kind takes the one kernel (16-row blocks): a prompt row is the bytes a decode row is.
 *  Refuses (-1) on a contract violation. */
int pulsar_nvfp4_rows_linear_launch(const void *w, const uint16_t *x_bf16, int rows, int in, int out, float *y,
                                    uint16_t *y_bf16, cudaStream_t stream);

// SPDX-License-Identifier: MIT
// The EXL3 dense-Linear arm (L251): one trellis-coded weight, M activation rows,
// a COMPLETE output -- y = svh * H128( W_hat^T H128(suh * x) ) per row, the
// format's whole factorization (src/engine/exl3_trellis.h).
//
// The weight is one [trellis | suh | svh] slice in the byte model of
// exl3_expert_layout(K, N, k2) -- the same slice a routed expert-projection is,
// so a checkpoint's tensors copy in verbatim.  The activation is the dense
// lane's A8 slot, exactly what the MXFP8 GEMV reads: E4M3 row-major [M][K] plus
// one ue8m0 byte per 32 at pulsar_mx_sfoff(row, k/32, pulsar_mx_kbp(K)).
//
// Where the two rotations go:
//   * INPUT (suh, then H128): the arm's first kernel applies them in f32 to the
//     slot's decoded rows, once per (row, 128-block), into its workspace.  That
//     is a multiplication by part of W, not an activation re-encode: the slot's
//     E4M3 bytes stay the one activation encoding.  It is not hoisted into the
//     producer because suh is per Linear and one activation feeds several
//     Linears (Qwen's DeltaNet in_proj_qkv / _z / _b / _a, an attention's
//     q / k / v): a pre-rotated slot per consumer would be several encodings of
//     one value, and E4M3 of H(suh * x) is a different quantization than the A8
//     contract's E4M3 of x.  The rotated rows are held as a power-of-two
//     prescale and an fp16 hi + lo pair (~22 significant bits) because that is
//     the operand the tensor cores take; the products with the (fp16-exact)
//     weight are exact, so the arm is f32-class -- graded against the host
//     authority in double at the same ~2e-7 an f32 FMA arm reaches.
//   * OUTPUT (H128, then svh): a dense Linear has no fold/sum epilogue, and the
//     output Hadamard mixes 128 outputs a GEMV CTA does not own together (a CTA
//     owns 32), so the arm's split-K GEMV writes unrotated f32 partials into the
//     workspace and its third kernel sums the splits in split order, rotates
//     each 128-block and scales by svh.
//
// Every row is bit-identical at every M: the split count depends on (K, N)
// only, every row block is 16 rows (the MMA's M; rows past M are zero), a
// row's k order, warp order and split order do not depend on its batchmates,
// and rows beyond 128 are further slabs of the same three launches -- decode
// and prefill are one arithmetic, as in the routed-expert arm.  Prefill rows
// are correct and slow: every 16-row block re-reads and re-decodes the weight
// (an E4M3-tile + FP8 MMA prefill arm is the one to build, L251 slice 2d).

#pragma once

#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

/** True when `k2` (rate in half-bit units) is a rate the dense arm
 *  instantiates: K = 2, 3, 4, 5. */
bool ds4_exl3_dense_rate_supported(int k2);

/** Workspace bytes ds4_exl3_dense_launch needs for (M, K, N): for one slab of
 *  min(M, 128) rows, the split partials (f32 [splits][rows][N]), the rotated
 *  rows' hi and lo planes (fp16 [rows][K] each) and their inverse prescales
 *  (f32 [rows][K/128]).  0 for a shape the arm refuses. */
size_t ds4_exl3_dense_workspace_bytes(int M, int K, int N);

/** The number of K splits the arm uses for (K, N) -- a function of the weight's
 *  shape alone, which is what keeps every row M-independent.  0 on a refused
 *  shape. */
int ds4_exl3_dense_splits(int K, int N);

/** y [M][N] f32 = the complete Linear.  `w` is the [trellis | suh | svh] slice
 *  (16-byte aligned), `xq` / `sx` the E4M3 slot of the M rows.  K % 128 == 0,
 *  N % 128 == 0, M >= 1, k2 a supported rate, `workspace` (16-byte aligned) at
 *  least ds4_exl3_dense_workspace_bytes(M, K, N).  Three kernels per 128-row
 *  slab on `stream`; refuses (returns -1) on any contract violation, -3 on a
 *  launch failure. */
int ds4_exl3_dense_launch(
    const void   * w,
    int            k2,
    const void   * xq,
    const void   * sx,
    float        * y,
    int            M,
    int            K,
    int            N,
    void         * workspace,
    size_t         workspace_bytes,
    cudaStream_t   stream);

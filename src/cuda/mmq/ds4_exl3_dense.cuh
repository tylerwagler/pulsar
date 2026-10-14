// SPDX-License-Identifier: MIT
// The EXL3 dense-Linear arm (L251): one trellis-coded weight, M activation rows,
// a COMPLETE output -- y = svh * H128( W_hat^T H128(suh * x) ) per row, the
// format's whole factorization (src/engine/exl3_trellis.h).
//
// The weight is one [trellis | suh | svh] slice in the byte model of
// exl3_expert_layout(K, N, k2) -- the same slice a routed expert-projection is,
// so a checkpoint's tensors copy in verbatim.  The activation is the block
// input's bf16 row -- the family's `x` -- and NOT an A8 slot (L251 / ac69748f:
// there is no E4M3 activation slot in this family).
//
// Where the two rotations go:
//   * INPUT (suh, then H128): the arm's kernel applies them in f32 to the
//     block input's bf16 rows, per (row, 128-block) of each CTA's k-range,
//     straight into the CTA's tensor-core operand tiles.
//     That is a multiplication by part of W, not an activation re-encode: the
//     bf16 row IS the one activation encoding, read as bf16 (rule 3).  It is not
//     hoisted into the producer because suh is per Linear and one activation
//     feeds several Linears (Qwen's DeltaNet in_proj_qkv / _z / _b / _a, an
//     attention's q / k / v): a pre-rotated copy per consumer would duplicate one
//     value several times, and it is not the value the producer emitted.  The
//     rotated rows are held as a power-of-two
//     prescale and an fp16 hi + lo pair (~22 significant bits) because that is
//     the operand the tensor cores take; the products with the (fp16-exact)
//     weight are exact, so the arm is f32-class -- graded against the host
//     authority in double at the same ~2e-7 an f32 FMA arm reaches.
//   * OUTPUT (H128, then svh): a dense Linear has no fold/sum epilogue, and the
//     output Hadamard mixes 128 outputs a GEMV CTA does not own together (a CTA
//     owns 32), so the arm's split-K GEMV writes unrotated f32 partials into the
//     workspace and the last CTA of each 128-output block to finish sums the
//     splits in split order, rotates the block and scales by svh.
//
// Every row is bit-identical at every M: the split count depends on (K, N)
// only, every row block is 16 rows (the MMA's M; rows past M are zero), a
// row's k order, warp order and split order do not depend on its batchmates;
// decode rows past 16 run 16-row blocks in turn (L288 C1).  Prompt chunks take
// the tensor-core prefill GEMM (qwen_exl3_dense_prefill.cu), which agrees to
// rounding.

#pragma once

#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

/** True when `k2` (rate in half-bit units) is a rate the dense arm
 *  instantiates: K = 2, 3, 4, 5. */
bool ds4_exl3_dense_rate_supported(int k2);

/** Workspace bytes ds4_exl3_dense_launch needs for (M, K, N): the split
 *  partials and the prep planes of min(M, 16) rows (decode rows past 16 reuse
 *  them a block at a time; a prompt chunk's prefill GEMM needs none).  0 for a
 *  shape the arm refuses. */
size_t ds4_exl3_dense_workspace_bytes(int M, int K, int N);

/** The number of K splits the arm uses for (K, N) -- a function of the weight's
 *  shape alone, which is what keeps every row M-independent.  0 on a refused
 *  shape. */
int ds4_exl3_dense_splits(int K, int N);

/** y [M][N] f32 = the complete Linear.  `w` is the [trellis | suh | svh] slice
 *  (16-byte aligned), `xb` the bf16 block input of the M rows (8-byte aligned,
 *  the uint2 row loads).  K % 128 == 0,
 *  N % 128 == 0, M >= 1, k2 a supported rate, `workspace` (16-byte aligned) at
 *  least ds4_exl3_dense_workspace_bytes(M, K, N).  Two kernels on `stream`
 *  (the prep, the GEMV with its epilogue); refuses (returns -1) on any contract violation, -3 on a
 *  launch failure. */
int ds4_exl3_dense_launch(
    const void   * w,
    int            k2,
    const void   * xb,
    float        * y,
    int            M,
    int            K,
    int            N,
    void         * workspace,
    size_t         workspace_bytes,
    cudaStream_t   stream,
    /* L266: a prompt chunk takes the prefill GEMM at every M (qwen_exl3_dense_prefill.cu), so a
     * prompt cut anywhere is byte-identical to one prefilled whole; false: decode rows, the GEMV at
     * every M (16-row blocks past 16, L288 C1) */
    bool           prompt = false);

/** L251: the dense arm's prompt chunks through a decode-once tensor-core GEMM: w is the dense layout
 *  (trellis | suh[K] | svh[N]); writes y [M][N] finished (output rotation and svh applied).
 *  ds4_exl3_dense_launch calls it for prompt chunks. */
int qwen_exl3_dense_prefill_launch(const void *w, int k2, const void *x_bf16, float *y, int M, int K, int N,
                                   uint64_t trellis_bytes, cudaStream_t stream);

/** The most projections one ds4_exl3_dense_multi_launch carries. */
#define DS4_EXL3_DENSE_MAX_PROJ 4

/** One projection of a multi launch: its [trellis | suh | svh] slice (16-byte aligned), rate, output width and
 *  output y [M][N] f32 (16-byte aligned). */
struct ds4_exl3_dense_proj {
    const void *w;
    int         k2;
    int         N;
    float      *y;
};

/** Workspace bytes ds4_exl3_dense_multi_launch needs for these projections at (M, K): the sum of each one's
 *  ds4_exl3_dense_workspace_bytes(M, K, N).  0 for a set the arm refuses. */
size_t ds4_exl3_dense_multi_workspace_bytes(const ds4_exl3_dense_proj *p, int n, int M, int K);

/** n (1..DS4_EXL3_DENSE_MAX_PROJ) projections of ONE input xb [M][K] bf16 (L288 exl3-dense: a GDN block's
 *  in_proj_qkv and _z, an attention's q / k / v / index, a shared expert's gate and up): one prep launch for all of
 *  them, then one GEMV launch per rate among them.  Each output is bit for bit what ds4_exl3_dense_launch writes for
 *  it alone.  Prompt chunks take the prefill GEMM, once per projection; decode rows past 16 run 16-row blocks in
 *  turn.  ds4_exl3_dense_launch is the
 *  n = 1 call. */
int ds4_exl3_dense_multi_launch(const ds4_exl3_dense_proj *p, int n, const void *xb, int M, int K, void *workspace,
                                size_t workspace_bytes, cudaStream_t stream, bool prompt = false);

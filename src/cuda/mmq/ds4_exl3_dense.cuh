// SPDX-License-Identifier: MIT
/** @file
 *  The EXL3 dense-Linear arm's contract: one trellis-coded weight against M bf16 rows, a complete output. */
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
 *  instantiates: exl3_arm_has_rate(EXL3_ARM_DENSE), K = 2, 3, 4, 5, 6, 8. */
/// @param k2 the rate in half-bit units (2 x bits per weight).
/// @return true when the arm instantiates the rate.
bool ds4_exl3_dense_rate_supported(int k2);

/** Workspace bytes ds4_exl3_dense_launch needs for (M, K, N): the split
 *  partials (at the stored width) and the prep planes of min(M, 16) rows
 *  (decode rows past 16 reuse them a block at a time; a prompt chunk's prefill
 *  GEMM needs none).  0 for a shape the arm refuses. */
/// @param M activation rows.
/// @param K input width (a multiple of 128).
/// @param N logical output width (a multiple of 4).
/// @return bytes, or 0 for a refused shape.
size_t ds4_exl3_dense_workspace_bytes(int M, int K, int N);

/** The number of K splits the arm uses for (K, N) -- a function of the weight's
 *  (stored) shape alone, which is what keeps every row M-independent.  0 on a
 *  refused shape. */
/// @param K input width (a multiple of 128).
/// @param N logical output width (a multiple of 4).
/// @return the split count, or 0 for a refused shape.
int ds4_exl3_dense_splits(int K, int N);

/** y [M][N] f32 = the complete Linear.  `w` is the [trellis | suh | svh] slice
 *  (16-byte aligned) in exl3_linear_layout(K, N, k2): N is the LOGICAL output
 *  width, stored at exl3_out_stored(N) (a 128 multiple, exllamav3's pad); the
 *  arm computes the stored width and writes the N logical columns, y's row
 *  width (L295 F7: a 201024-wide head stored 201088).  `xb` the bf16 block
 *  input of the M rows (8-byte aligned, the uint2 row loads).  K % 128 == 0,
 *  N % 4 == 0, M >= 1, k2 a supported rate, `workspace` (16-byte aligned) at
 *  least ds4_exl3_dense_workspace_bytes(M, K, N).  Two kernels on `stream`
 *  (the prep, the GEMV with its epilogue); refuses (returns -1) on any contract violation, -3 on a
 *  launch failure. */
/// @param w the [trellis | suh | svh] slice, device, 16-byte aligned.
/// @param k2 the rate in half-bit units (2 x bits per weight).
/// @param xb the bf16 input rows, [M][K], 8-byte aligned.
/// @param y out: f32 [M][N].
/// @param M activation rows.
/// @param K input width (a multiple of 128).
/// @param N logical output width (a multiple of 4).
/// @param workspace device workspace, 16-byte aligned.
/// @param workspace_bytes bytes of workspace.
/// @param stream the stream.
/// @param prompt the rows are a prompt chunk (the prefill GEMM), else decode rows (the GEMV).
/// @return 0, -1 refused, -3 launch failure.
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
 *  (trellis | suh[K] | svh[stored N], exl3_linear_layout); writes y [M][N] finished (output rotation and svh
 *  applied), N the logical width.
 *  ds4_exl3_dense_launch calls it for prompt chunks. */
/// @param w the [trellis | suh | svh] slice, device, 16-byte aligned.
/// @param k2 the rate in half-bit units (2 x bits per weight).
/// @param x_bf16 the bf16 input rows, [M][K].
/// @param y out: f32 [M][N].
/// @param M activation rows.
/// @param K input width (a multiple of 128).
/// @param N logical output width (a multiple of 4).
/// @param trellis_bytes bytes of the slice's trellis plane (suh follows it).
/// @param stream the stream.
/// @return 0, -1 refused, -3 launch failure.
int qwen_exl3_dense_prefill_launch(const void *w, int k2, const void *x_bf16, float *y, int M, int K, int N,
                                   uint64_t trellis_bytes, cudaStream_t stream);

/** The most projections one ds4_exl3_dense_multi_launch carries. */
#define DS4_EXL3_DENSE_MAX_PROJ 4

/** One projection of a multi launch: its [trellis | suh | svh] slice (16-byte aligned), rate, output width and
 *  output: y [M][N] f32 (16-byte aligned), or (L296 dec2, decode rows only) NULL y and yb [M][N] bf16 (8-byte
 *  aligned) -- the epilogue emits the bf16 rows the f32 output rounds to (round to nearest even), for a consumer
 *  that reads bf16; a prompt chunk refuses yb. */
struct ds4_exl3_dense_proj {
    /// The [trellis | suh | svh] slice, device, 16-byte aligned (exl3_linear_layout).
    const void *w;
    /// Rate: 2 x bits per weight (4, 6, 8, 10, 12 or 16).
    int         k2;
    /// Logical output width (a multiple of 4; stored at its 128 multiple).
    int         N;
    /// f32 output [M][N], 16-byte aligned, or NULL when yb is set.
    float      *y;
    /// bf16 output [M][N] (decode rows only), 8-byte aligned, or NULL.
    uint16_t   *yb;
};

/** Workspace bytes ds4_exl3_dense_multi_launch needs for these projections at (M, K): the sum of each one's
 *  ds4_exl3_dense_workspace_bytes(M, K, N).  0 for a set the arm refuses. */
/// @param p the projections, [n].
/// @param n projections (1 .. DS4_EXL3_DENSE_MAX_PROJ).
/// @param M activation rows.
/// @param K input width (a multiple of 128).
/// @return bytes, or 0 for a refused set.
size_t ds4_exl3_dense_multi_workspace_bytes(const ds4_exl3_dense_proj *p, int n, int M, int K);

/** n (1..DS4_EXL3_DENSE_MAX_PROJ) projections of ONE input xb [M][K] bf16 (L288 exl3-dense: a GDN block's
 *  in_proj_qkv and _z, an attention's q / k / v / index, a shared expert's gate and up): one prep launch for all of
 *  them, then one GEMV launch per rate among them.  Each output is bit for bit what ds4_exl3_dense_launch writes for
 *  it alone.  Prompt chunks take the prefill GEMM, once per projection; decode rows past 16 run 16-row blocks in
 *  turn.  ds4_exl3_dense_launch is the
 *  n = 1 call. */
/// @param p the projections, [n].
/// @param n projections (1 .. DS4_EXL3_DENSE_MAX_PROJ).
/// @param xb the bf16 input rows, [M][K], 8-byte aligned.
/// @param M activation rows.
/// @param K input width (a multiple of 128).
/// @param workspace device workspace, 16-byte aligned.
/// @param workspace_bytes bytes of workspace.
/// @param stream the stream.
/// @param prompt the rows are a prompt chunk (the prefill GEMM), else decode rows (the GEMV).
/// @return 0, -1 refused, -3 launch failure.
int ds4_exl3_dense_multi_launch(const ds4_exl3_dense_proj *p, int n, const void *xb, int M, int K, void *workspace,
                                size_t workspace_bytes, cudaStream_t stream, bool prompt = false);

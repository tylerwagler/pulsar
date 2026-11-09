/** @file
 *  The f16 rows arm (L321): an IEEE f16 Linear over raw bf16 rows on the fp16 tensor cores. */
/* The f16 rows arm (L321): a Linear stored IEEE f16 [out][in] (PULSAR_TENSOR_F16, the Turing lane's dense format)
 * over raw bf16 rows, on the fp16 tensor cores (sm_75+).
 *
 * THE ARITHMETIC (the EXL3 prompt arm's prep, without its rotation):
 *   per row: P = the power of two that puts the row's largest |x| just under 2^14; x16 = f16(x * 2^P) -- EXACT for
 *            every value within 2^27 of the row's largest (a bf16 value's 8 significant bits fit f16's 11, and the
 *            scaled exponent stays in f16's normal range), so the operand is the bf16 row, scaled;
 *   y = (x16 . W^T) * 2^-P   f32 accumulate and output: the fixed-algorithm GEMM (pulsar_cuda_lt_fixed.h) -- PROMPT
 *            rows on the algorithm chosen at the prefill chunk, DECODE rows in blocks of PULSAR_GPU_MNEUTRAL_ROWS_MAX on
 *            the one chosen there -- so a row's bytes never depend on how many rows share its call.
 * Or the operand rows ARE f16 (L321: their producer wrote the fp16 consumer's format -- pulsar_rows_out): the GEMM on
 * them as they are, no prescale (pulsar_f16_rows_linear_h_launch; their range is the producer's).
 * No bias here: a family with biases adds them where it emits (pulsar_rows_bias_emit_launch / res32_add). */
#ifndef PULSAR_CUDA_F16_ROWS_H
#define PULSAR_CUDA_F16_ROWS_H

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

/** Workspace bytes pulsar_f16_rows_linear_launch needs for `rows` rows of width `in`. */
/// @param rows rows.
/// @param in input width.
/// @return bytes (the scaled f16 rows, then the per-row inverse scales, each 256-byte padded); 0 for a non-positive shape.
size_t pulsar_f16_rows_workspace_bytes(int rows, int in);

/** y [rows][out] f32 = the bf16 rows x [rows][in] times the f16 weight w [out][in] (device, 16-byte aligned; in and
 *  out multiples of 8).  @return 0, -1 refused (named on stderr), -3 launch failure. */
/// @param w the weight, f16 [out][in].
/// @param x_bf16 the rows, bf16 [rows][in].
/// @param rows rows.
/// @param in input width.
/// @param out output width.
/// @param y the output, f32 [rows][out].
/// @param prompt the rows are a prompt chunk (else decode blocks of PULSAR_GPU_MNEUTRAL_ROWS_MAX).
/// @param ws device workspace.
/// @param ws_bytes bytes of ws (>= pulsar_f16_rows_workspace_bytes).
/// @param stream the stream.
int pulsar_f16_rows_linear_launch(const void *w, const uint16_t *x_bf16, int rows, int in, int out, float *y,
                                  bool prompt, void *ws, size_t ws_bytes, cudaStream_t stream);

/** y [rows][out] f32 = the f16 rows x [rows][in] times the f16 weight w [out][in], on the fixed-algorithm GEMM of the
 *  row kind; no workspace.  @return 0, -1 refused (named on stderr), -3 launch failure. */
/// @param w the weight, f16 [out][in].
/// @param x_f16 the rows, f16 [rows][in].
/// @param rows rows.
/// @param in input width.
/// @param out output width.
/// @param y the output, f32 [rows][out].
/// @param prompt the rows are a prompt chunk.
/// @param stream the stream.
int pulsar_f16_rows_linear_h_launch(const void *w, const uint16_t *x_f16, int rows, int in, int out, float *y,
                                    bool prompt, cudaStream_t stream);

/** L338: one projection's runtime LoRA: its device pair at the logical widths and its scale (the container's f16
 *  pairs: A [r][in], B [out][r], 16-byte aligned; r, in and out multiples of 8). */
typedef struct pulsar_rows_lora {
    const void *A;    ///< device f16 [r][in]: lora_A
    const void *B;    ///< device f16 [out][r]: lora_B
    uint32_t r;       ///< the rank
    uint32_t in;      ///< the input width (<= the input pitch)
    uint32_t out;     ///< the output width (<= the output pitch)
    float scale;      ///< alpha / r
} pulsar_rows_lora;

/** Rows pulsar_rows_lora_launch runs at a time (a call walks its rows in blocks of this many -- the fixed-algorithm
 *  GEMM's bytes do not depend on the rows beside a row, so the blocks are the call whole), and so the rows its
 *  workspace holds at most. */
#define PULSAR_ROWS_LORA_BLOCK 256

/** Workspace bytes pulsar_rows_lora_launch needs for `rows` rows (at most PULSAR_ROWS_LORA_BLOCK count) of the widest
 *  LoRA (in, r, out). */
/// @param rows rows.
/// @param in   the widest input width.
/// @param r    the widest rank.
/// @param out  the widest output width.
/// @return bytes (each piece 256-byte padded); 0 for a non-positive shape.
size_t pulsar_rows_lora_workspace_bytes(int rows, int in, int r, int out);

/** L338: a runtime LoRA on a linear's f32 output, on the fp16 tensor cores --
 *    y[t][o] += scale * sum_j B[o][j] * (sum_k A[j][k] * x[t][k])     for t < n_rows marked (rows NULL = every row)
 *  THE ARITHMETIC: x16 = f16(x * 2^P) per row (the arm's prep: exact for the bf16 values); T = x16 . A^T (f32, the
 *  fixed-algorithm GEMM); T16 = f16(T * 2^Q) per row (Q puts the row's largest |T| just under 2^14: no overflow); U =
 *  T16 . B^T (f32, the same GEMM); y += (scale * 2^-P * 2^-Q) * U -- the factor a product of powers of two times scale,
 *  one f32 rounding into y.  So the error is f16's on T (the pairs are f16 as stored); PROMPT rows on the prompt class's
 *  algorithm, DECODE rows in blocks of PULSAR_GPU_MNEUTRAL_ROWS_MAX on the decode class's: a row's bytes never depend
 *  on how many rows share its call.  A row not marked is computed but not added (its bytes are the base linear's);
 *  columns of y past `out` and of x past `in` are neither written nor read.
 *  @return 0, -1 refused (named on stderr), -3 launch failure. */
/// @param x        device bf16 input rows (ld_x apart; the linear's input)
/// @param ld_x     x's row pitch in elements (>= in)
/// @param n_rows   rows (>= 1)
/// @param l        the LoRA
/// @param y        device f32 output rows (ld_y apart), accumulated into
/// @param ld_y     y's row pitch in elements (>= out)
/// @param rows     device u8 [n_rows]: the rows the LoRA applies to (nonzero), or NULL = every row
/// @param prompt   the rows are a prompt chunk (else decode rows)
/// @param ws       device workspace
/// @param ws_bytes its bytes (>= pulsar_rows_lora_workspace_bytes(n_rows, l->in, l->r, l->out))
/// @param stream   the stream
int pulsar_rows_lora_launch(const uint16_t *x, uint64_t ld_x, int n_rows, const pulsar_rows_lora *l, float *y,
                            uint64_t ld_y, const uint8_t *rows, bool prompt, void *ws, size_t ws_bytes,
                            cudaStream_t stream);

#endif

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
size_t pulsar_f16_rows_workspace_bytes(int rows, int in);

/** y [rows][out] f32 = the bf16 rows x [rows][in] times the f16 weight w [out][in] (device, 16-byte aligned; in and
 *  out multiples of 8).  @return 0, -1 refused (named on stderr), -3 launch failure. */
int pulsar_f16_rows_linear_launch(const void *w, const uint16_t *x_bf16, int rows, int in, int out, float *y,
                                  bool prompt, void *ws, size_t ws_bytes, cudaStream_t stream);

/** y [rows][out] f32 = the f16 rows x [rows][in] times the f16 weight w [out][in], on the fixed-algorithm GEMM of the
 *  row kind; no workspace.  @return 0, -1 refused (named on stderr), -3 launch failure. */
int pulsar_f16_rows_linear_h_launch(const void *w, const uint16_t *x_f16, int rows, int in, int out, float *y,
                                    bool prompt, cudaStream_t stream);

#endif

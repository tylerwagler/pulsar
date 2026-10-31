/** @file
 *  Bidirectional fused attention over f16 rows: the entry to the vendored CUTLASS memory-efficient attention (Sm75 instance). */
/* Bidirectional fused attention over f16 rows (L321): CUTLASS's memory-efficient attention (examples/41, xformers'
 * kernel; vendored at the CUTLASS pin in src/cuda/fmha/, VENDOR.md), the Sm75 instance -- fp16 operands, f32
 * accumulate and softmax, run on the fp16 tensor cores of every arch from Turing up.
 *
 * One sequence of n_rows tokens, every row attending every row (no mask): y[r][h] = softmax_k(scale q[r][h] . k[k][h])
 * v[k][h], per head h, written as f16 rows.  q / k / v are f16 rows read in place at any leading dimension (a fused
 * qkv Linear's output: q, k, v its column blocks) with the heads head_dim apart; head_dim a multiple of 8, at most
 * 128 (Nemotron Parse's RADIO ViT: 16 heads of 72 over a page's ~13k patches; the shared GQA block's tiles are 64 and
 * 128 wide and its operands bf16).
 *
 * Not bit-exact to anything: the kernel's own blocked online softmax, graded against a double reference by
 * tests/fmha_gate (and the model by its own grade). */
#ifndef PULSAR_CUDA_FMHA_H
#define PULSAR_CUDA_FMHA_H

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** y = attention(q, k, v) for one sequence of n_rows rows and n_heads heads of head_dim; every pointer 16-byte
 *  aligned, every leading dimension a multiple of 8 elements.  @return 0, -1 refused (named on stderr), -3 launch
 *  failure. */
/// @param q        device f16 query rows, head h at column h * head_dim
/// @param ld_q     q row pitch in elements (a multiple of 8)
/// @param k        device f16 key rows, same head layout
/// @param ld_k     k row pitch in elements
/// @param v        device f16 value rows, same head layout
/// @param ld_v     v row pitch in elements
/// @param n_rows   tokens in the sequence (every row attends every row)
/// @param n_heads  heads
/// @param head_dim per-head width (a multiple of 8, at most 128)
/// @param scale    softmax scale on q . k
/// @param y        device f16 output rows, same head layout
/// @param ld_y     y row pitch in elements
/// @param stream   stream to launch on
int pulsar_fmha_f16_launch(const uint16_t *q, uint64_t ld_q, const uint16_t *k, uint64_t ld_k, const uint16_t *v,
                           uint64_t ld_v, uint32_t n_rows, uint32_t n_heads, uint32_t head_dim, float scale, uint16_t *y,
                           uint64_t ld_y, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

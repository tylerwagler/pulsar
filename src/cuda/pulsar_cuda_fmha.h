/* Bidirectional fused attention over f16 rows (L321) and bf16 rows (L331): CUTLASS's memory-efficient attention (examples/41, xformers'
 * kernel; vendored at the CUTLASS pin in src/cuda/fmha/, VENDOR.md): the Sm75 instance -- fp16 operands, f32
 * accumulate and softmax, run on the fp16 tensor cores of every arch from Turing up -- and (L331) the Sm80 instance
 * of the same kernel for bf16 operands (sm_80+; refused by name below it).
 *
 * One sequence of n_rows tokens, every row attending every row (no mask): y[r][h] = softmax_k(scale q[r][h] . k[k][h])
 * v[k][h], per head h, written as f16 rows.  q / k / v are f16 rows read in place at any leading dimension (a fused
 * qkv Linear's output: q, k, v its column blocks) with the heads head_dim apart; head_dim a multiple of 8, at most
 * 128 (Nemotron Parse's RADIO ViT: 16 heads of 72 over a page's ~13k patches, f16; Nemotron 3.5 Super VL's C-RADIO
 * ViT-H: 16 heads of 80 over up to 13322 rows, bf16 -- the shared GQA block's tiles are 64 and 128 wide).
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
int pulsar_fmha_f16_launch(const uint16_t *q, uint64_t ld_q, const uint16_t *k, uint64_t ld_k, const uint16_t *v,
                           uint64_t ld_v, uint32_t n_rows, uint32_t n_heads, uint32_t head_dim, float scale, uint16_t *y,
                           uint64_t ld_y, cudaStream_t stream);

/** L331: the same over bf16 rows (q / k / v and y bf16), on the bf16 tensor cores: -1 refused below sm_80. */
int pulsar_fmha_bf16_launch(const uint16_t *q, uint64_t ld_q, const uint16_t *k, uint64_t ld_k, const uint16_t *v,
                            uint64_t ld_v, uint32_t n_rows, uint32_t n_heads, uint32_t head_dim, float scale, uint16_t *y,
                            uint64_t ld_y, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

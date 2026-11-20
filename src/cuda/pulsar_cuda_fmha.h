/** @file
 *  Bidirectional fused attention over f16 and bf16 rows: the entry to the vendored CUTLASS memory-efficient attention
 *  (Sm75 f16 and Sm80 bf16 instances). */
/* Bidirectional fused attention over f16 rows (L321) and bf16 rows (L331): CUTLASS's memory-efficient attention
 * (examples/41, xformers' kernel; vendored at the CUTLASS pin in src/cuda/fmha/, VENDOR.md): the Sm75 instance -- fp16
 * operands, f32 accumulate and softmax, run on the fp16 tensor cores of every arch from Turing up -- and (L331) the
 * Sm80 instance of the same kernel for bf16 operands (sm_80+; refused by name below it).
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

/** L331: the same over bf16 rows (q / k / v and y bf16), on the bf16 tensor cores: -1 refused below sm_80. */
/// @param q        device bf16 query rows, head h at column h * head_dim
/// @param ld_q     q row pitch in elements (a multiple of 8)
/// @param k        device bf16 key rows, same head layout
/// @param ld_k     k row pitch in elements
/// @param v        device bf16 value rows, same head layout
/// @param ld_v     v row pitch in elements
/// @param n_rows   tokens in the sequence (every row attends every row)
/// @param n_heads  heads
/// @param head_dim per-head width (a multiple of 8, at most 128)
/// @param scale    softmax scale on q . k
/// @param y        device bf16 output rows, same head layout
/// @param ld_y     y row pitch in elements
/// @param stream   stream to launch on
/// @return 0, -1 refused (named on stderr; also below sm_80), -3 launch failure
int pulsar_fmha_bf16_launch(const uint16_t *q, uint64_t ld_q, const uint16_t *k, uint64_t ld_k, const uint16_t *v,
                            uint64_t ld_v, uint32_t n_rows, uint32_t n_heads, uint32_t head_dim, float scale, uint16_t *y,
                            uint64_t ld_y, cudaStream_t stream);

/** L446: causal grouped-query attention at head_dim 512 over f16 rows, on the fp16 tensor cores (the Sm75 instance of the
 *  same kernel at kMaxK 512, 32 queries x 128 keys a block: the output accumulates in an f32 buffer across its value
 *  iterations).  The Turing prompt arm of the GQA block's global layers (pulsar_cuda_attn_gqa.cu): one sequence's run
 *  of n_rows query rows at positions n_keys - n_rows .. n_keys - 1, each attending keys 0 .. its own position (the
 *  bottom-right causal mask).  A row's keys are walked in 128-key blocks from position 0, so its bytes do not depend on
 *  which other rows share the call.
 *  Layouts (f16, elements):
 *    q    [n_rows][n_kv * G][512]  query head h = g * G + j of KV head g
 *    k, v [n_kv][n_keys][512]      one KV head's rows contiguous (read by all G of its query heads: stride 0)
 *    y    [n_kv][n_rows][G][512]   f16 output;  acc the same shape in f32 (the kernel's accumulator)
 *  @return 0, -1 refused (named on stderr), -3 launch failure. */
/// @param q      device f16 query rows
/// @param k      device f16 key rows
/// @param v      device f16 value rows
/// @param n_kv   KV heads
/// @param G      query heads per KV head
/// @param n_rows query rows (the run's last n_rows positions)
/// @param n_keys keys (the run's last position + 1)
/// @param scale  softmax scale on q . k
/// @param y      device f16 output
/// @param acc    device f32 accumulator, as y
/// @param stream stream to launch on
int pulsar_fmha_f16_gqa_causal512_launch(const uint16_t *q, const uint16_t *k, const uint16_t *v, uint32_t n_kv,
                                         uint32_t G, uint32_t n_rows, uint32_t n_keys, float scale, uint16_t *y,
                                         float *acc, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

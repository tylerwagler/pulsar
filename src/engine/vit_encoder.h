/* vit_encoder.h -- the timm / RADIO ViT encoder as ONE shared block (L331): the blocks Nemotron Parse's C-RADIOv4-SO400M
 * (L321) and Nemotron 3.5 Super VL's C-RADIO ViT-H/16 both run, family-neutral.
 *
 * THE BLOCK (timm's Block, RADIO's RadioLayer: no LayerScale, no drop path), over an f32 RESIDUAL res [n][width]:
 *     xn = LN1(res);  qkv = xn Wqkv + b;  a = MHA(q, k, v)  (bidirectional over all n rows, 1/sqrt(head_dim));
 *     res += a Wproj + b;  xn = LN2(res);  m = GELU(xn Wfc1 + b);  res += m Wfc2 + b
 * LayerNorm eps the caller's; GELU the exact (erf) one.  The residual is f32 because RADIO's late layers grow a few
 * very-high-norm tokens and a bf16 residual drops the small updates every other row gets (Parse measured it: vit_out
 * row cosine down to 0.08).
 *
 * THE OPERAND FORMAT is the caller's and decides every consumer (rule 3, the producers emit it):
 *   PULSAR_ROWS_OUT_F16   f16 rows, f16 weights: the f16 rows arm (pulsar_rows_linear_f16_launch) and the f16 FMHA
 *                         (Parse, whose checkpoint and Turing lane are f16);
 *   PULSAR_ROWS_OUT_BF16  bf16 rows, bf16 weights: the bf16 rows front door (pulsar_rows_linear_launch: the BF16 rows
 *                         arm) and the bf16 FMHA (Nemotron Super's tower, stored BF16).
 * Weights are bound by the family (role records `vit_*`, block `vit`); norms and biases f32.
 *
 * The call takes the SEEDED residual (the caller's prefix tokens, positions and patch embedding are its own) and
 * leaves the last block's residual in res; nothing is allocated here.  Linears run in row chunks of `chunk` (a row's
 * bytes do not depend on its chunk). */
#ifndef PULSAR_VIT_ENCODER_H
#define PULSAR_VIT_ENCODER_H

#include "cuda/pulsar_cuda_rowops.h"

/** One block's tensors (the family binds them). */
typedef struct {
    /** @var pulsar_vit_block::norm1
     *  vit_norm1: LN1's weight, f32 [width]
     *  @var pulsar_vit_block::norm1_b
     *  vit_norm1_bias: LN1's bias, f32 [width]
     *  @var pulsar_vit_block::qkv
     *  vit_qkv: the fused q | k | v Linear [width -> 3 heads head_dim]
     *  @var pulsar_vit_block::qkv_b
     *  vit_qkv_bias: its bias, f32 [3 heads head_dim]
     *  @var pulsar_vit_block::proj
     *  vit_proj: the attention output Linear [heads head_dim -> width]
     *  @var pulsar_vit_block::proj_b
     *  vit_proj_bias: its bias, f32 [width]
     *  @var pulsar_vit_block::norm2
     *  vit_norm2: LN2's weight, f32 [width]
     *  @var pulsar_vit_block::norm2_b
     *  vit_norm2_bias: LN2's bias, f32 [width]
     *  @var pulsar_vit_block::fc1
     *  vit_fc1: the MLP's first Linear [width -> mlp]
     *  @var pulsar_vit_block::fc1_b
     *  vit_fc1_bias: its bias, f32 [mlp]
     *  @var pulsar_vit_block::fc2
     *  vit_fc2: the MLP's second Linear [mlp -> width]
     *  @var pulsar_vit_block::fc2_b
     *  vit_fc2_bias: its bias, f32 [width] */
    pulsar_tensor *norm1, *norm1_b, *qkv, *qkv_b, *proj, *proj_b, *norm2, *norm2_b, *fc1, *fc1_b, *fc2, *fc2_b;
} pulsar_vit_block;

/** The encoder's geometry. */
typedef struct {
    /** @var pulsar_vit_dims::width
     *  the residual width
     *  @var pulsar_vit_dims::heads
     *  attention heads
     *  @var pulsar_vit_dims::head_dim
     *  a head's width
     *  @var pulsar_vit_dims::mlp
     *  the MLP's inner width (fc1's output)
     *  @var pulsar_vit_dims::depth
     *  blocks run (`blocks[0..depth)`) */
    uint32_t width, heads, head_dim, mlp, depth;
    /// the LayerNorms' epsilon
    float eps;
} pulsar_vit_dims;

/** The device buffers: res f32 [n][width] (seeded); xn [n][width], qkv [n][3 heads head_dim], att [n][heads head_dim]
 *  in the operand format; mlp [chunk][mlp] in it; stage f32 [chunk][max(3 heads head_dim, mlp)]; lws / lws_bytes the
 *  rows front door's workspace (the bf16 operand only; pulsar_rows_linear_workspace_bytes). */
typedef struct {
    /// the residual, f32 [n][width], seeded by the caller; holds the last block's residual on return
    float *res;
    /** @var pulsar_vit_bufs::xn
     *  the LayerNorm output [n][width], operand format
     *  @var pulsar_vit_bufs::qkv
     *  the fused q | k | v rows [n][3 heads head_dim], operand format
     *  @var pulsar_vit_bufs::att
     *  the attention output [n][heads head_dim], operand format
     *  @var pulsar_vit_bufs::mlp
     *  one chunk's GELU(fc1) rows [chunk][mlp], operand format */
    uint16_t *xn, *qkv, *att, *mlp;
    /// the Linears' f32 output, one chunk: [chunk][max(3 heads head_dim, mlp)]
    float *stage;
    /// the rows front door's workspace (the bf16 operand only)
    void *lws;
    /// the size of `lws` in bytes
    uint64_t lws_bytes;
    /// the Linears' row chunk
    uint32_t chunk;
} pulsar_vit_bufs;

/** Optional per-block hooks (the caller's GPU phases, its stage dumps): before block il, after it.  NULL = none. */
typedef struct {
    /// called before block il runs (NULL = none)
    void (*before_block)(void *ud, uint32_t il);
    /// called after block il ran (NULL = none)
    void (*after_block)(void *ud, uint32_t il);
    /// the hooks' first argument
    void *ud;
} pulsar_vit_hooks;

/** The blocks over the n seeded rows of res: LN1 of block 0, then `depth` blocks.  false + `err` (said) on a refusal. */
bool pulsar_vit_encode(const pulsar_model *m, const pulsar_vit_dims &d, const pulsar_vit_block *blocks,
                       pulsar_rows_out fmt, const pulsar_vit_bufs &b, uint32_t n, const pulsar_vit_hooks *hooks,
                       char *err, size_t errlen);

#endif /* PULSAR_VIT_ENCODER_H */

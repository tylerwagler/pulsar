/* vit_encoder.cpp -- the shared timm / RADIO ViT encoder (vit_encoder.h): Parse's tower loop (L321) made the block
 * both RADIO towers run, its operand format a parameter (L331). */
#include "pulsar_engine_internal.h"
#include "vit_encoder.h"
#include "cuda/pulsar_cuda_fmha.h"
#include "cuda/pulsar_cuda_qwen.h"   /* the rows front door's launchers (family-neutral) */

#include <math.h>

namespace {

struct vit_ctx {
    const pulsar_model *m;
    pulsar_rows_out fmt;
    const pulsar_vit_bufs &b;
    char *err;
    size_t errlen;
    bool fail(const char *what) {
        if (err) snprintf(err, errlen, "vit: %s", what);
        return false;
    }
    const float *fp(const pulsar_tensor *t, const char *what) {
        return (const float *)pulsar_weight_device_ptr(m, t, what);
    }
    /* stage [rows][out] f32 = x [rows][in] times t, on the operand format's arm */
    bool lin(const pulsar_tensor *t, int in, int out, const uint16_t *x, uint32_t rows, const char *what) {
        pulsar_rows_linear l;
        if (!pulsar_linear_rows_ref(m, t, in, out, true, what, &l)) return fail(what);
        const int rc = fmt == PULSAR_ROWS_OUT_F16
                           ? pulsar_rows_linear_f16_launch(&l, x, (int)rows, b.stage, 0)
                           : pulsar_rows_linear_launch(&l, x, (int)rows, b.stage, b.lws, b.lws_bytes, 0);
        return rc == 0 || fail(what);
    }
};

}  // namespace

bool pulsar_vit_encode(const pulsar_model *m, const pulsar_vit_dims &d, const pulsar_vit_block *blocks,
                       pulsar_rows_out fmt, const pulsar_vit_bufs &b, uint32_t n, const pulsar_vit_hooks *hooks,
                       char *err, size_t errlen) {
    vit_ctx c{m, fmt, b, err, errlen};
    const uint32_t D = d.width, HD = d.head_dim, HS = d.heads * HD, M = d.mlp, C = b.chunk;
    float *res = b.res;
    uint16_t *xn = b.xn, *qkv = b.qkv, *att = b.att, *mlp = b.mlp;
    if (pulsar_rows_layernorm32_launch(res, true, D, NULL, n, D, c.fp(blocks[0].norm1, "vit norm1"),
                                       c.fp(blocks[0].norm1_b, "vit norm1"), d.eps, xn, D, fmt, NULL, 0, 0) != 0)
        return c.fail("the first LayerNorm refused");
    const float scale = 1.0f / sqrtf((float)HD);
    for (uint32_t il = 0; il < d.depth; il++) {
        if (hooks && hooks->before_block) hooks->before_block(hooks->ud, il);
        const pulsar_vit_block &B = blocks[il];
        for (uint32_t r0 = 0; r0 < n; r0 += C) {
            const uint32_t rows = n - r0 < C ? n - r0 : C;
            if (!c.lin(B.qkv, (int)D, (int)(3u * HS), xn + (size_t)r0 * D, rows, "vit qkv") ||
                pulsar_rows_bias_emit_launch(b.stage, 3u * HS, rows, 3u * HS, c.fp(B.qkv_b, "vit qkv bias"),
                                             PULSAR_ROWS_ACT_NONE, qkv + (size_t)r0 * 3u * HS, 3u * HS, fmt, 0) != 0)
                return c.fail("the qkv refused");
        }
        const int arc = fmt == PULSAR_ROWS_OUT_F16
                            ? pulsar_fmha_f16_launch(qkv, 3u * HS, qkv + HS, 3u * HS, qkv + 2u * HS, 3u * HS, n, d.heads,
                                                     HD, scale, att, HS, 0)
                            : pulsar_fmha_bf16_launch(qkv, 3u * HS, qkv + HS, 3u * HS, qkv + 2u * HS, 3u * HS, n,
                                                      d.heads, HD, scale, att, HS, 0);
        if (arc != 0) return c.fail("the attention refused");
        const bool last = il + 1 == d.depth;
        const float *n1 = last ? NULL : c.fp(blocks[il + 1].norm1, "vit norm1");
        const float *n1b = last ? NULL : c.fp(blocks[il + 1].norm1_b, "vit norm1");
        for (uint32_t r0 = 0; r0 < n; r0 += C) {
            const uint32_t rows = n - r0 < C ? n - r0 : C;
            float *rc = res + (size_t)r0 * D;
            uint16_t *xc = xn + (size_t)r0 * D;
            if (!c.lin(B.proj, (int)HS, (int)D, att + (size_t)r0 * HS, rows, "vit proj") ||
                pulsar_rows_res32_add_launch(rc, D, b.stage, D, c.fp(B.proj_b, "vit proj bias"), rows, D,
                                             c.fp(B.norm2, "vit norm2"), c.fp(B.norm2_b, "vit norm2"), d.eps, xc, D,
                                             fmt, 0) != 0 ||
                !c.lin(B.fc1, (int)D, (int)M, xc, rows, "vit fc1") ||
                pulsar_rows_bias_emit_launch(b.stage, M, rows, M, c.fp(B.fc1_b, "vit fc1 bias"), PULSAR_ROWS_ACT_GELU,
                                             mlp, M, fmt, 0) != 0 ||
                !c.lin(B.fc2, (int)M, (int)D, mlp, rows, "vit fc2") ||
                pulsar_rows_res32_add_launch(rc, D, b.stage, D, c.fp(B.fc2_b, "vit fc2 bias"), rows, D, n1, n1b, d.eps,
                                             last ? NULL : xc, D, fmt, 0) != 0)
                return c.fail("a block refused");
        }
        if (hooks && hooks->after_block) hooks->after_block(hooks->ud, il);
    }
    return true;
}

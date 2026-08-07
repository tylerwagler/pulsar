#!/usr/bin/env python3
"""Add the IQ2_XXS single-tensor D2R dispatch to the vendored ds4_mmq.cu.

Upstream's single-tensor moe impl already has a D2R fast path, but only for
GGML_TYPE_Q2_K.  An IQ2 routed DOWN therefore fell through to stock mul_mat_q
(measured 33.24 ms/layer, against Entrpi's 9.78 ms Q2_K D2R) even when handed an
aligned artifact.  This inserts the IQ2 twin of that block, calling the
single-tensor launcher appended to ds4_mmq_d2r.cu.

Idempotent: exits 0 with a message if the dispatch is already present.
"""
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "src/cuda/mmq/ds4_mmq.cu"

ANCHOR = """    if (type == GGML_TYPE_Q2_K && x_soa != nullptr && d2r_enabled() &&
        K % 256 == 0 && M % 2 == 0 && ne_get_rows >= d2r_min_cols()) {"""

NEW = """    /* pulsar (plan 41b): IQ2_XXS twin of the Q2_K block below.  Our v5mx down
     * tensors are IQ2, not Q2_K, so without this an aligned IQ2 down still ran
     * stock mul_mat_q.  Same gating, same scratch contract; the launcher pins
     * the pair kernel's leg to 0 so it computes one tensor. */
    if (type == GGML_TYPE_IQ2_XXS && x_soa != nullptr && d2r_enabled() &&
        d2r_iq2_enabled() && K % 256 == 0 && ne_get_rows >= d2r_min_cols()) {
        static int d2r_iq2s_cc = -1;
        static int d2r_iq2s_avail = 0;
        if (d2r_iq2s_cc != cc) {
            d2r_iq2s_cc = cc;
            d2r_iq2s_avail = ds4_mmq_iq2_xxs_moe_d2r_available(cc) ? 1 : 0;
        }
        if (d2r_iq2s_avail) {
            const size_t w_bytes =
                ds4_mmq_iq2_xxs_moe_d2r_pair_scratch_bytes(ne_get_rows, n_experts);
            if (w_bytes != 0) {
                ggml_cuda_pool_alloc<char> d2r_work(ctx->pool(), w_bytes);
                const int rc = ds4_mmq_iq2_xxs_moe_d2r_single_launch(
                    x_soa, soa_blocks, src1_q8_1.get(), ids_dst.get(), expert_bounds.get(),
                    out_f32, M, K, ne_get_rows, n_experts, d2r_work.get(), w_bytes, stream);
                if (rc == 0) {
                    return 0;
                }
            }
        }
    }

"""

src = open(PATH).read()
if "d2r_iq2s_avail" in src:
    print("already patched")
    sys.exit(0)
if ANCHOR not in src:
    print("ANCHOR not found -- vendored ds4_mmq.cu differs from expected", file=sys.stderr)
    sys.exit(1)
open(PATH, "w").write(src.replace(ANCHOR, NEW + ANCHOR, 1))
print("IQ2 single-tensor D2R dispatch inserted")

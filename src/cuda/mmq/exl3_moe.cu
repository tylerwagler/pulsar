// SPDX-License-Identifier: MIT
// The EXL3 routed MoE's one front door (exl3_moe.cuh has the contract and the dataflow).

#include "exl3_moe.cuh"
#include "exl3_moe_epilogue.cuh"
#include "ds4_exl3_dense.cuh"
#include "ds4_mmid.cuh"
#include "ds4_mmq.h"
#include "../pulsar_cuda_scratch.h"
#include "engine/exl3_trellis.h"

#include <cstdio>

namespace {

constexpr uint64_t a256(uint64_t b) { return (b + 255u) & ~(uint64_t)255u; }

bool act_ok(const exl3_moe_act &a) {
    return a.x && (a.format == EXL3_MOE_ACT_BF16_ROWS ||
                   (a.format == EXL3_MOE_ACT_E4M3_SLOT && a.sf && a.kbp > 0));
}

/* One projection of the schedule, through the arm the rule picks: a prompt chunk's grouped GEMM (once per slice,
 * each rotating its input by its own suh) writes the caller's z; the decode GEMM (a pair's two slices in one
 * launch) leaves split partials in the workspace, which *z points the fold / sum at. */
int project(int kind, const void *table, const void *up_table, int k2, const exl3_moe_act &act,
            const int32_t *ids_dst, const int32_t *ids_src, const int32_t *bounds, float *out, float *out_up,
            int M, int K, int64_t pairs, int E, bool gemm, void *tc_ws, size_t tc_bytes, exl3_moe_partials *z,
            cudaStream_t stream) {
    const bool rotate = kind != EXL3_ARM_DOWN;
    if (!gemm)
        return exl3_moe_decode_launch(table, kind == EXL3_ARM_PAIR ? up_table : nullptr, k2, rotate, act, ids_dst,
                                      ids_src, bounds, M, K, pairs, E, tc_ws, tc_bytes, z, stream);
    int rc = exl3_moe_prefill_launch(table, k2, rotate, act, ids_dst, ids_src, bounds, out, M, K, pairs, E, tc_ws,
                                     tc_bytes, stream);
    if (rc == 0 && kind == EXL3_ARM_PAIR)
        rc = exl3_moe_prefill_launch(up_table, k2, true, act, ids_dst, ids_src, bounds, out_up, M, K, pairs, E,
                                     tc_ws, tc_bytes, stream);
    *z = exl3_moe_partials{{out, out_up}, 1, 0};
    return rc;
}

} // namespace

size_t exl3_moe_shared_ws_bytes(int T, int in_dim, int mid, int out_dim) {
    const size_t gu = 2 * ds4_exl3_dense_workspace_bytes(T, in_dim, mid), d = ds4_exl3_dense_workspace_bytes(T, mid, out_dim);
    return gu > d ? gu : d;
}

int exl3_moe_routed_launch(const exl3_moe_stacks &w, const int32_t *sel, const float *wts, int T, int top_k,
                           float clamp, const exl3_moe_act &x, const exl3_moe_bufs &b, float *out,
                           uint32_t *nf_flag, uint32_t nf_code, bool prompt, cudaStream_t stream,
                           const exl3_moe_shared *shared) {
    const char *tag = "exl3_moe_routed_launch";
    const bool fused = w.gate_up != nullptr;
    const int gu_kind = fused ? EXL3_ARM_GATE_UP_FUSED : EXL3_ARM_PAIR;
    if (fused == (w.gate != nullptr) || (!fused && !w.up) || !w.down || !sel || !wts || !out || !nf_flag ||
        !b.gate_z || (!fused && !b.up_z) || !b.down_z || !act_ok(x) || !act_ok(b.mid) || T <= 0 || top_k <= 0 ||
        w.n_experts <= 0 || w.in_dim <= 0 || w.mid_dim <= 0 || w.out_dim <= 0) {
        fprintf(stderr, "%s: the stacks must be ONE of a fused gate_up or a gate + up pair, with a down, and every "
                        "buffer present -- refusing\n", tag);
        return -1;
    }
    if (!exl3_arm_has_rate(gu_kind, w.k2_gate_up) || !exl3_arm_has_rate(EXL3_ARM_DOWN, w.k2_down) ||
        w.in_dim % EXL3_HAD_BLOCK || w.mid_dim % EXL3_HAD_BLOCK || w.out_dim % EXL3_HAD_BLOCK) {
        fprintf(stderr, "%s: %s K=%g / down K=%g at %d -> %d -> %d has no instance (128-multiples, "
                        "exl3_arm_has_rate) -- refusing\n", tag, fused ? "fused gate_up" : "gate / up", w.k2_gate_up / 2.0,
                w.k2_down / 2.0, w.in_dim, w.mid_dim, w.out_dim);
        return -1;
    }
    const exl3_moe_shared *sh = shared;
    if (sh && (!sh->gate || !sh->up || !sh->down || !sh->yg || !sh->yu || !sh->ys || !sh->h || !sh->ws || sh->mid <= 0 ||
               x.format != EXL3_MOE_ACT_BF16_ROWS)) {
        fprintf(stderr, "%s: a riding shared expert needs its three slices, its buffers and a bf16-rows activation "
                        "(the dense arm reads bf16; act format %d) -- refusing\n", tag, x.format);
        return -1;
    }
    static int mmq_ready = -1;   /* the sort's device-info singleton */
    if (mmq_ready < 0) {
        int dev = 0;
        (void)cudaGetDevice(&dev);
        mmq_ready = ds4_mmq_init(dev) == 0 ? 1 : 0;
    }
    if (!mmq_ready) {
        fprintf(stderr, "%s: the MMQ drivers are unavailable on this device -- refusing\n", tag);
        return -1;
    }
    const int64_t pairs = (int64_t)T * top_k;
    if (pairs > INT32_MAX) return -1;
    const int E = w.n_experts;
    const bool gemm = prompt;   /* the row kind is the one rule: a prompt chunk the grouped GEMM, decode rows the decode arm */
    size_t tc_bytes = 0;
    {
        /* gate / up's workspace, then the down's over it: the fold has read gate / up's z before the down's PREP */
        const size_t a = gemm ? exl3_moe_prefill_ws_bytes(pairs, w.in_dim, E)
                              : exl3_moe_decode_ws_bytes(fused ? 1 : 2, pairs, w.in_dim, fused ? 2 * w.mid_dim : w.mid_dim);
        const size_t d = gemm ? exl3_moe_prefill_ws_bytes(pairs, w.mid_dim, E)
                              : exl3_moe_decode_ws_bytes(1, pairs, w.mid_dim, w.out_dim);
        tc_bytes = a > d ? a : d;
    }
    const uint64_t ids_bytes = (uint64_t)pairs * sizeof(int32_t), eb_bytes = (uint64_t)(E + 1) * sizeof(int32_t);
    cuda_arena ar;
    if (!cuda_arena_begin_slot(&ar, CUDA_SCRATCH_MMQ, 2 * a256(ids_bytes) + a256(eb_bytes) + a256(tc_bytes), tag)) {
        fprintf(stderr, "%s: scratch reservation failed\n", tag);
        return -1;
    }
    int32_t *ids_src = (int32_t *)cuda_arena_take(&ar, ids_bytes, 256);
    int32_t *ids_dst = (int32_t *)cuda_arena_take(&ar, ids_bytes, 256);
    int32_t *bounds = (int32_t *)cuda_arena_take(&ar, eb_bytes, 256);
    void *tc_ws = cuda_arena_take(&ar, tc_bytes, 256);
    if (!tc_ws) return -1;   /* take() latches */

    /* a riding shared expert's gate and up: one dense launch when they share a rate */
    if (sh) {
        int rc = sh->k2_gate == sh->k2_up
            ? ds4_exl3_dense_pair_launch(sh->gate, sh->up, sh->k2_gate, x.x, sh->yg, sh->yu, T, w.in_dim, sh->mid, sh->ws,
                                         sh->ws_bytes, stream, prompt)
            : ds4_exl3_dense_launch(sh->gate, sh->k2_gate, x.x, sh->yg, T, w.in_dim, sh->mid, sh->ws, sh->ws_bytes, stream,
                                    prompt);
        if (rc == 0 && sh->k2_gate != sh->k2_up)
            rc = ds4_exl3_dense_launch(sh->up, sh->k2_up, x.x, sh->yu, T, w.in_dim, sh->mid, sh->ws, sh->ws_bytes, stream,
                                       prompt);
        if (rc) { fprintf(stderr, "%s: the shared expert's gate / up declined (rc=%d) -- no fallback\n", tag, rc); return -1; }
    }
    /* the schedule, once: every consumer stops at the schedule's ends (expert_bounds[0], expert_bounds[E]), so
     * the unwritten head of a schedule with dropped picks is never read */
    ds4_launch_mm_ids_helper(sel, ids_src, ids_dst, bounds, E, T, top_k, /*nchannels_y=*/1, /*si1=*/top_k,
                             /*sis1=*/1, stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: mm_ids_helper failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    const exl3_moe_stacks &s = w;
    exl3_moe_partials gp{}, dp{};
    int rc = project(gu_kind, fused ? s.gate_up : s.gate, fused ? nullptr : s.up, s.k2_gate_up, x, ids_dst, ids_src,
                     bounds, b.gate_z, fused ? nullptr : b.up_z, fused ? 2 * s.mid_dim : s.mid_dim, s.in_dim, pairs, E,
                     gemm, tc_ws, tc_bytes, &gp, stream);
    if (rc) { fprintf(stderr, "%s: gate / up declined (rc=%d) -- no fallback\n", tag, rc); return -1; }
    float *gz = gp.z[0], *uz = gp.z[1];
    const bool mid_e4m3 = b.mid.format == EXL3_MOE_ACT_E4M3_SLOT;
    void *mq = mid_e4m3 ? const_cast<void *>(b.mid.x) : nullptr, *msf = mid_e4m3 ? const_cast<void *>(b.mid.sf) : nullptr;
    void *mb = mid_e4m3 ? nullptr : const_cast<void *>(b.mid.x);
    const exl3_moe_shared_swiglu shs{sh ? sh->yg : nullptr, sh ? sh->yu : nullptr, sh ? sh->h : nullptr, T,
                                     sh ? sh->mid : 0};
    rc = fused ? ds4_exl3_moe_fold_fused_launch(gz, sel, wts, s.gate_up, s.down, s.in_dim, s.mid_dim, pairs, clamp, mq,
                                                msf, b.mid.kbp, stream, mb, sh ? &shs : nullptr, gp.splits, gp.split_stride)
               : ds4_exl3_moe_fold_launch(gz, uz, sel, wts, s.gate, s.up, s.down, s.in_dim, s.mid_dim, pairs, clamp, mq,
                                          msf, b.mid.kbp, stream, mb, sh ? &shs : nullptr, gp.splits, gp.split_stride);
    if (rc) return -1;
    if (sh && ds4_exl3_dense_launch(sh->down, sh->k2_down, sh->h, sh->ys, T, sh->mid, w.out_dim, sh->ws, sh->ws_bytes,
                                    stream, prompt) != 0) {
        fprintf(stderr, "%s: the shared expert's down declined -- no fallback\n", tag);
        return -1;
    }
    /* the down: its source row is the pair (the fold wrote one mid row per pair) */
    rc = project(EXL3_ARM_DOWN, s.down, nullptr, s.k2_down, b.mid, ids_dst, ids_dst, bounds, b.down_z, nullptr,
                 s.out_dim, s.mid_dim, pairs, E, gemm, tc_ws, tc_bytes, &dp, stream);
    if (rc) { fprintf(stderr, "%s: down declined (rc=%d) -- no fallback\n", tag, rc); return -1; }
    rc = ds4_exl3_moe_sum_launch(out, dp.z[0], sel, s.down, s.mid_dim, s.out_dim, top_k, T, nf_flag, nf_code, stream,
                                 sh ? sh->ys : nullptr, sh ? sh->scale : nullptr, dp.splits, dp.split_stride);
    return rc ? -1 : 0;
}

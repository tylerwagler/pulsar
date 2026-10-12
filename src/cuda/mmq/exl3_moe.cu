// SPDX-License-Identifier: MIT
// The EXL3 routed MoE's one front door (exl3_moe.cuh has the contract and the dataflow).

#include "exl3_moe.cuh"
#include "exl3_moe_epilogue.cuh"
#include "ds4_exl3_dense.cuh"
#include "ds4_mmid.cuh"
#include "ds4_mmq.h"
#include "../pulsar_cuda_scratch.h"
#include "cuda/pulsar_cuda_mx.cuh"
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

/* The riding shared expert's lane (L288 exl3-dense): its own stream, one per (host thread, device, caller
 * stream).  exl3_moe_shared_fork puts the WHOLE shared expert on it -- gate / up (one dense multi launch), its
 * SwiGLU, its down -- forked off the caller's stream at x, before the family's router: those latency-bound
 * launches (a 2 MB and a 1 MB projection, 1-2 blocks per CTA) then run beside the router, the schedule and the
 * routed GEMMs instead of after them, and exl3_moe_routed_launch joins the lane right before the sum that adds
 * the shared output.  Only the schedule changes: every value is the arithmetic it had (the SwiGLU rows are the
 * fold's own expression; the dense arm's epilogue counters are per stream already, counter_slice).  A lane holds
 * up to kLanePending forks the caller has not joined yet (one per row-kind segment of a step, each over its own
 * rows' buffers).  The lanes live as long as their thread. */
constexpr int kLanePending = 4;
struct shared_lane {
    int dev;
    cudaStream_t caller, s;
    cudaEvent_t fork;
    cudaEvent_t done[kLanePending];
    const float *ys[kLanePending];   ///< the pending forks' outputs, oldest first
    int n;
};

shared_lane *shared_lane_for(cudaStream_t caller) {
    thread_local shared_lane t_lanes[8];
    thread_local int t_n = 0;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return nullptr;
    for (int i = 0; i < t_n; ++i)
        if (t_lanes[i].dev == dev && t_lanes[i].caller == caller) return &t_lanes[i];
    if (t_n == 8) return nullptr;
    shared_lane l{};
    l.dev = dev;
    l.caller = caller;
    if (cudaStreamCreateWithFlags(&l.s, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&l.fork, cudaEventDisableTiming) != cudaSuccess)
        return nullptr;
    for (cudaEvent_t &e : l.done)
        if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) return nullptr;
    t_lanes[t_n] = l;
    static bool announced = false;
    if (!__atomic_exchange_n(&announced, true, __ATOMIC_RELAXED))
        fprintf(stderr, "pulsar: EXL3 shared expert = gate / up + SwiGLU + down on its own stream, forked at x ahead of "
                        "the router, joined at the routed sum\n");
    return &t_lanes[t_n++];
}

/* The shared expert's SwiGLU rows for its down: h = silu(g) * u, bf16 -- the expression the fold emitted them with
 * (pulsar_swiglu_elem, weight 1, no clamp), now the lane's producer of h. */
__global__ void __launch_bounds__(256)
exl3_shared_swiglu_kernel(const float *__restrict__ g, const float *__restrict__ u, __nv_bfloat16 *__restrict__ h,
                          int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = __float2bfloat16(pulsar_swiglu_elem(g[i], u[i], 1.0f, 0.0f));
}

} // namespace

size_t exl3_moe_shared_ws_bytes(int T, int in_dim, int mid, int out_dim) {
    const ds4_exl3_dense_proj p[2] = {{nullptr, 0, mid, nullptr}, {nullptr, 0, mid, nullptr}};
    const size_t gu = ds4_exl3_dense_multi_workspace_bytes(p, 2, T, in_dim), d = ds4_exl3_dense_workspace_bytes(T, mid, out_dim);
    return gu > d ? gu : d;
}

int exl3_moe_shared_fork(const exl3_moe_shared &sh, const exl3_moe_act &x, int T, int in_dim, int out_dim, bool prompt,
                         cudaStream_t stream) {
    const char *tag = "exl3_moe_shared_fork";
    if (!sh.gate || !sh.up || !sh.down || !sh.yg || !sh.yu || !sh.ys || !sh.h || !sh.ws || sh.mid <= 0 || T <= 0 ||
        in_dim <= 0 || out_dim <= 0 || x.format != EXL3_MOE_ACT_BF16_ROWS || !x.x) {
        fprintf(stderr, "%s: a shared expert needs its three slices, its buffers and a bf16-rows activation (the dense "
                        "arm reads bf16; act format %d) -- refusing\n", tag, x.format);
        return -1;
    }
    shared_lane *lane = shared_lane_for(stream);
    if (!lane || lane->n == kLanePending) {
        fprintf(stderr, "%s: no lane, or %d forks already pending on it -- refusing\n", tag, lane ? lane->n : 0);
        return -1;
    }
    if (cudaEventRecord(lane->fork, stream) != cudaSuccess || cudaStreamWaitEvent(lane->s, lane->fork, 0) != cudaSuccess)
        return -3;
    const ds4_exl3_dense_proj gu[2] = {{sh.gate, sh.k2_gate, sh.mid, sh.yg}, {sh.up, sh.k2_up, sh.mid, sh.yu}};
    int rc = ds4_exl3_dense_multi_launch(gu, 2, x.x, T, in_dim, sh.ws, sh.ws_bytes, lane->s, prompt);
    if (rc == 0) {
        const int64_t n = (int64_t)T * sh.mid;
        exl3_shared_swiglu_kernel<<<(unsigned)((n + 255) / 256), 256, 0, lane->s>>>(sh.yg, sh.yu, sh.h, n);
        rc = cudaGetLastError() == cudaSuccess ? 0 : -3;
    }
    if (rc == 0)
        rc = ds4_exl3_dense_launch(sh.down, sh.k2_down, sh.h, sh.ys, T, sh.mid, out_dim, sh.ws, sh.ws_bytes, lane->s,
                                   prompt);
    if (rc == 0 && cudaEventRecord(lane->done[lane->n], lane->s) != cudaSuccess) rc = -3;
    if (rc) {
        fprintf(stderr, "%s: the shared expert's launches declined (rc=%d) -- no fallback\n", tag, rc);
        (void)cudaStreamSynchronize(lane->s);   /* nothing of a refused fork outlives the call */
        return -1;
    }
    lane->ys[lane->n++] = sh.ys;
    return 0;
}

namespace {

/* The caller's stream waits for the OLDEST pending fork, which must be the one whose output is `ys`. */
int shared_join(const float *ys, cudaStream_t stream, const char *tag) {
    shared_lane *lane = shared_lane_for(stream);
    if (!lane || lane->n == 0 || lane->ys[0] != ys) {
        fprintf(stderr, "%s: a riding shared expert was not forked first (exl3_moe_shared_fork) on this stream -- "
                        "refusing\n", tag);
        return -1;
    }
    if (cudaStreamWaitEvent(stream, lane->done[0], 0) != cudaSuccess) return -3;
    /* pop: the events rotate with their entries */
    const cudaEvent_t e = lane->done[0];
    for (int i = 1; i < lane->n; ++i) {
        lane->done[i - 1] = lane->done[i];
        lane->ys[i - 1] = lane->ys[i];
    }
    lane->done[--lane->n] = e;
    return 0;
}

} // namespace

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
    if (sh && x.format != EXL3_MOE_ACT_BF16_ROWS) {
        fprintf(stderr, "%s: a riding shared expert needs a bf16-rows activation (act format %d) -- refusing\n", tag,
                x.format);
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
    rc = fused ? ds4_exl3_moe_fold_fused_launch(gz, sel, wts, s.gate_up, s.down, s.in_dim, s.mid_dim, pairs, clamp, mq,
                                                msf, b.mid.kbp, stream, mb, gp.splits, gp.split_stride)
               : ds4_exl3_moe_fold_launch(gz, uz, sel, wts, s.gate, s.up, s.down, s.in_dim, s.mid_dim, pairs, clamp, mq,
                                          msf, b.mid.kbp, stream, mb, gp.splits, gp.split_stride);
    if (rc) return -1;
    /* the down: its source row is the pair (the fold wrote one mid row per pair) */
    rc = project(EXL3_ARM_DOWN, s.down, nullptr, s.k2_down, b.mid, ids_dst, ids_dst, bounds, b.down_z, nullptr,
                 s.out_dim, s.mid_dim, pairs, E, gemm, tc_ws, tc_bytes, &dp, stream);
    if (rc) { fprintf(stderr, "%s: down declined (rc=%d) -- no fallback\n", tag, rc); return -1; }
    /* the sum adds the shared expert's output: its fork joins here */
    if (sh && shared_join(sh->ys, stream, tag) != 0) return -1;
    rc = ds4_exl3_moe_sum_launch(out, dp.z[0], sel, s.down, s.mid_dim, s.out_dim, top_k, T, nf_flag, nf_code, stream,
                                 sh ? sh->ys : nullptr, sh ? sh->scale : nullptr, dp.splits, dp.split_stride);
    return rc ? -1 : 0;
}

// SPDX-License-Identifier: MIT
// The EXL3 routed MoE's ONE front door, for every family (task #44; it was two launchers -- Qwen's bf16-rows
// door in pulsar_cuda_qwen_moe.cu and DeepSeek's slot door in pulsar_cuda_moe.cu -- each with its own copy
// of the sort / gate-up / fold / down / sum plumbing, over five MMQ entry points).
//
//   sort      mm_ids_helper ONCE: the expert-sorted schedule (ids_src = token, ids_dst = pair, expert_bounds).
//             The down's schedule is the same sort with the pair as its source row (ids_src = ids_dst), so it
//             is not recomputed.
//   gate/up   fp16 tensor-core arms (exl3_moe_prefill.cuh), chosen by the row kind alone: decode rows the
//             split-K decode GEMM (task #46; a gate + up PAIR in one launch), a prompt chunk the grouped
//             GEMM; a FUSED gate_up slice or a gate + up PAIR, over the activation as its producer emitted
//             it (bf16 rows or the E4M3 slot)
//   fold      svh, H128, SwiGLU (clamp; the route weight folded in), the down's input rotation, emitted in the
//             format the down reads -- the caller's choice of mid (bf16 rows or an E4M3 slot), the producer
//             rule
//   down      the same two arms over the mid
//   sum       the slot-ordered sum with the down's output rotation; a non-finite value records nf_code
//
// A shared expert (of a block's N, in order) whose three Linears are EXL3 dense slices RIDES these launches (keyed on that format, not on
// a family; exl3_moe_shared): the family forks it first (exl3_moe_shared_fork, ahead of its router) onto its own
// stream -- gate and up as one dense multi launch, its SwiGLU rows, its down -- and the routed launch joins it
// right before the sum, which adds its scaled output after the routed sum: the order the separate launches had,
// each value the arithmetic it had alone (L288 exl3-dense: those latency-bound launches now run beside the router,
// the schedule and the routed GEMMs instead of after them).  A shared expert in another format, or none, is the
// family's to run.  The dense arm reads bf16 rows, so a riding shared expert needs a bf16 activation: over
// an E4M3 slot the call refuses.
//
// Shapes come from the call (the model's), never from a family's constants.  A pick < 0 (another rank's
// expert under expert parallelism, or the router's NaN path) is not in the schedule and adds nothing.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

#include "exl3_moe_prefill.cuh"   /* exl3_moe_act */

/** One layer's routed EXL3 stacks: exl3_expert_table() pointer pairs over the experts this rank holds. */
struct exl3_moe_stacks {
    const void *gate_up;            ///< the FUSED [in -> 2 mid] slices (gate rows, then up rows), or NULL
    const void *gate, *up;          ///< the PAIR form, or NULL
    const void *down;
    int k2_gate_up, k2_down;        ///< rates in half-bit units (the pair's gate and up share one)
    int n_experts;                  ///< experts the tables hold (picks are local indices < n_experts)
    int in_dim, mid_dim, out_dim;
};

/** The caller's buffers.  gate_z [pairs][mid] (fused: [pairs][2 mid]), up_z [pairs][mid] (pair form only),
 *  down_z [pairs][out]; mid = the fold's output in the format the down reads (x = the bf16 rows or the E4M3
 *  slot [pairs][mid], sf / kbp its ue8m0 plane). */
struct exl3_moe_bufs {
    float *gate_z, *up_z, *down_z;
    exl3_moe_act mid;
};

/** A shared expert riding the routed launches: three EXL3 dense slices ([trellis | suh | svh], 16-byte aligned) at
 *  in_dim -> mid -> out_dim, and its buffers.  `scale` [T] multiplies its output (a sigmoid gate), or NULL. */
struct exl3_moe_shared {
    const void *gate, *up, *down;
    int k2_gate, k2_up, k2_down;
    int mid;
    const float *scale;
    float *yg, *yu, *ys;            ///< [T][mid], [T][mid], [T][out_dim] f32
    __nv_bfloat16 *h;               ///< [T][mid] the SwiGLU rows its down reads
    void *ws;                       ///< the dense arm's workspace:
    size_t ws_bytes;                ///< exl3_moe_shared_ws_bytes
};
/** The dense-arm workspace a riding shared expert needs for T rows. */
size_t exl3_moe_shared_ws_bytes(int T, int in_dim, int mid, int out_dim);

/** Fork the shared expert of T rows of x (bf16 rows, in_dim wide) off `stream` onto its lane: ys [T][out_dim] f32 =
 *  down(swiglu(gate x, up x)) (unscaled; yg / yu / h hold the intermediates).  The exl3_moe_routed_launch on the same
 *  stream that is given the same `sh` joins it (forks are joined oldest first, at most 8 pending; give each pending
 *  fork its own buffers -- the workspace may be shared).  `prompt`: the rows' kind, as the routed launch's.  Refuses
 *  (-1) on a contract violation; a refused fork leaves nothing running. */
int exl3_moe_shared_fork(const exl3_moe_shared &sh, const exl3_moe_act &x, int T, int in_dim, int out_dim, bool prompt,
                         cudaStream_t stream);

/** out [T][out_dim] f32 = sum over each row's top_k picks (sel [T][top_k], wts [T][top_k]) of
 *  w * down(swiglu(gate x, up x)), x the activation as its producer emitted it, plus each riding shared expert's
 *  scale * ys, shared[0] first, when `shared` [n_shared] is given -- each forked first by exl3_moe_shared_fork, in
 *  that order (refused otherwise).  `prompt`: a prompt chunk
 *  (the prefill GEMM at every width, writing gate_z / up_z / down_z); otherwise decode rows (the decode GEMM
 *  at every width, whose split partials live in the scratch: b's z buffers are not written).  Scratch (the
 *  schedule, the operand planes, the decode partials) is the MMQ arena slot's.  Refuses (-1) on any contract violation; nothing is launched past a refusal. */
int exl3_moe_routed_launch(const exl3_moe_stacks &w, const int32_t *sel, const float *wts, int T, int top_k,
                           float clamp, const exl3_moe_act &x, const exl3_moe_bufs &b, float *out,
                           uint32_t *nf_flag, uint32_t nf_code, bool prompt, cudaStream_t stream,
                           const exl3_moe_shared *shared = nullptr, int n_shared = 0);

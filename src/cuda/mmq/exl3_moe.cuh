// SPDX-License-Identifier: MIT
/** @file
 *  The EXL3 routed MoE's one front door, every family: contract and dataflow. */
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

/** One layer's routed EXL3 stacks: exl3_expert_table() pointer pairs over the experts this rank holds.  Three forms:
 *  FUSED (gate_up), PAIR (gate + up) -- both SwiGLU -- and (L296) UP-ONLY (up alone, gate and gate_up NULL): an
 *  ungated expert, act = PULSAR_FFN_ACT_RELU2 (pulsar_cuda_qwen.h), h = relu(up x)^2. */
struct exl3_moe_stacks {
    const void *gate_up;            ///< the FUSED [in -> 2 mid] slices (gate rows, then up rows), or NULL
    /// The PAIR form's gate slices [in -> mid], or NULL (FUSED and UP-ONLY).
    const void *gate, *up;          ///< the PAIR form (or up alone: the UP-ONLY form), or NULL
    /// The down slices [mid -> out].
    const void *down;
    /// The gate / up rate in half-bit units (fused gate_up, the pair's shared rate, or up-only's up).
    int k2_gate_up, k2_down;        ///< rates in half-bit units (the pair's gate and up share one; up-only: the up's)
    int n_experts;                  ///< experts the tables hold (picks are local indices < n_experts)
    /** @var mid_dim
     *  the expert's intermediate width (a multiple of 128) */
    /** @var out_dim
     *  the output width (a multiple of 128) */
    /// The input width (a multiple of 128).
    int in_dim, mid_dim, out_dim;
    int act;                        ///< pulsar_ffn_act: 0 SWIGLU (fused / pair), 1 RELU2 (up-only); 0 when unnamed
};

/** The caller's buffers.  gate_z [pairs][mid] (fused: [pairs][2 mid]), up_z [pairs][mid] (pair form only),
 *  down_z [pairs][out]; mid = the fold's output in the format the down reads (x = the bf16 rows or the E4M3
 *  slot [pairs][mid], sf / kbp its ue8m0 plane). */
struct exl3_moe_bufs {
    /** @var up_z
     *  the up projection's z f32 [pairs][mid] (pair form), or NULL */
    /** @var down_z
     *  the down projection's z f32 [pairs][out] */
    /// The gate projection's z f32 [pairs][mid] (fused: [pairs][2 mid]).
    float *gate_z, *up_z, *down_z;
    /// The fold's output, the down's input, in the format the down reads.
    exl3_moe_act mid;
};

/** A shared expert riding the routed launches: three EXL3 dense slices ([trellis | suh | svh], 16-byte aligned) at
 *  in_dim -> mid -> out_dim, and its buffers.  `scale` [T] multiplies its output (a sigmoid gate), or NULL. */
struct exl3_moe_shared {
    /** @var up
     *  its up slice [in_dim -> mid] */
    /** @var down
     *  its down slice [mid -> out_dim] */
    /// Its gate slice [in_dim -> mid].
    const void *gate, *up, *down;
    /** @var k2_up
     *  the up slice's rate, half-bit units */
    /** @var k2_down
     *  the down slice's rate, half-bit units */
    /// The gate slice's rate, half-bit units.
    int k2_gate, k2_up, k2_down;
    /// Its intermediate width.
    int mid;
    /// Per-row output scale f32 [T] (a sigmoid gate), or NULL for 1.
    const float *scale;
    /** @var yu
     *  the up output f32 [T][mid] */
    /// The gate output f32 [T][mid].
    float *yg, *yu, *ys;            ///< [T][mid], [T][mid], [T][out_dim] f32
    __nv_bfloat16 *h;               ///< [T][mid] the SwiGLU rows its down reads
    void *ws;                       ///< the dense arm's workspace:
    size_t ws_bytes;                ///< exl3_moe_shared_ws_bytes
};
/** The dense-arm workspace a riding shared expert needs for T rows. */
/// @param T rows.
/// @param in_dim the block input width.
/// @param mid the shared expert's intermediate width.
/// @param out_dim the block output width (a multiple of 128).
/// @return bytes.
size_t exl3_moe_shared_ws_bytes(int T, int in_dim, int mid, int out_dim);

/** Fork the shared expert of T rows of x (bf16 rows, in_dim wide) off `stream` onto its lane: ys [T][out_dim] f32 =
 *  down(swiglu(gate x, up x)) (unscaled; yg / yu / h hold the intermediates).  The exl3_moe_routed_launch on the same
 *  stream that is given the same `sh` joins it (forks are joined oldest first, at most 8 pending; give each pending
 *  fork its own buffers -- the workspace may be shared).  `prompt`: the rows' kind, as the routed launch's.  Refuses
 *  (-1) on a contract violation; a refused fork leaves nothing running. */
/// @param sh the shared expert and its buffers.
/// @param x the activation as its producer emitted it (bf16 rows for a riding shared expert).
/// @param T rows.
/// @param in_dim the block input width.
/// @param out_dim the block output width (a multiple of 128).
/// @param prompt the rows are a prompt chunk (else decode rows).
/// @param stream the stream.
/// @return 0, -1 refused (nothing left running), -3 on a stream / event failure.
int exl3_moe_shared_fork(const exl3_moe_shared &sh, const exl3_moe_act &x, int T, int in_dim, int out_dim, bool prompt,
                         cudaStream_t stream);

/** out [T][out_dim] f32 = sum over each row's top_k picks (sel [T][top_k], wts [T][top_k]) of
 *  w * down(swiglu(gate x, up x)) (up-only: w * down(relu(up x)^2), b.up_z unused), x the activation as its producer emitted it, plus each riding shared expert's
 *  scale * ys, shared[0] first, when `shared` [n_shared] is given -- each forked first by exl3_moe_shared_fork, in
 *  that order (refused otherwise).  `prompt`: a prompt chunk
 *  (the prefill GEMM at every width, writing gate_z / up_z / down_z); otherwise decode rows (the decode GEMM
 *  at every width, whose split partials live in the scratch: b's z buffers are not written).  Scratch (the
 *  schedule, the operand planes, the decode partials) is the MMQ arena slot's.  Refuses (-1) on any contract violation; nothing is launched past a refusal. */
/// @param w the routed stacks.
/// @param sel the routes, i32 [T][top_k].
/// @param wts the route weights, f32 [T][top_k].
/// @param T rows.
/// @param top_k picks per row.
/// @param clamp the SwiGLU clamp.
/// @param x the activation as its producer emitted it (bf16 rows for a riding shared expert).
/// @param b the caller's z and mid buffers.
/// @param out out: f32 [T][out_dim].
/// @param nf_flag the non-finite flag (first writer wins).
/// @param nf_code the code written to it.
/// @param prompt the rows are a prompt chunk (else decode rows).
/// @param stream the stream.
/// @param shared riding shared experts added after the routed sum, or NULL.
/// @param n_shared entries in shared.
/// @return 0, -1 refused, -3 launch failure.
int exl3_moe_routed_launch(const exl3_moe_stacks &w, const int32_t *sel, const float *wts, int T, int top_k,
                           float clamp, const exl3_moe_act &x, const exl3_moe_bufs &b, float *out,
                           uint32_t *nf_flag, uint32_t nf_code, bool prompt, cudaStream_t stream,
                           const exl3_moe_shared *shared = nullptr, int n_shared = 0);

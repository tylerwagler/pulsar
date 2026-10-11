// SPDX-License-Identifier: MIT
// The EXL3 routed MoE's ONE front door, for every family (task #44; it was two launchers -- Qwen's bf16-rows
// door in pulsar_cuda_qwen_moe.cu and DeepSeek's slot door in pulsar_cuda_moe.cu -- each with its own copy
// of the sort / gate-up / fold / down / sum plumbing, over five MMQ entry points).
//
//   sort      mm_ids_helper ONCE: the expert-sorted schedule (ids_src = token, ids_dst = pair, expert_bounds).
//             The down's schedule is the same sort with the pair as its source row (ids_src = ids_dst), so it
//             is not recomputed.
//   gate/up   the trellis GEMV (ds4_exl3_gemv.cuh) at decode widths, the grouped tensor-core GEMM
//             (exl3_moe_prefill.cuh) for a prompt chunk -- exl3_moe_prefill_takes is the one rule; a FUSED
//             gate_up slice or a gate + up PAIR, over the activation as its producer emitted it (bf16 rows or
//             the E4M3 slot)
//   fold      svh, H128, SwiGLU (clamp; the route weight folded in), the down's input rotation, emitted in the
//             format the down reads -- the caller's choice of mid (bf16 rows or an E4M3 slot), the producer
//             rule
//   down      the same two arms over the mid
//   sum       the slot-ordered sum with the down's output rotation; a non-finite value records nf_code
//
// A shared expert whose three Linears are EXL3 dense slices RIDES these launches (keyed on that format, not on
// a family; exl3_moe_shared): its gate and up are one dense launch (ds4_exl3_dense_pair_launch) ahead of the
// routed gate / up, its SwiGLU is emitted in the fold's launch, its down is one dense launch ahead of the
// routed down, and its scaled output is added in the sum's launch -- after the routed sum, the order the
// separate launches had, each value the arithmetic it had alone.  A shared expert in another format, or none,
// is the family's to run.  The dense arm reads bf16 rows, so a riding shared expert needs a bf16 activation:
// over an E4M3 slot the call refuses.
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

/** out [T][out_dim] f32 = sum over each row's top_k picks (sel [T][top_k], wts [T][top_k]) of
 *  w * down(swiglu(gate x, up x)), x the activation as its producer emitted it, plus a riding shared expert's
 *  scale * down(swiglu(gate x, up x)) when `shared` is given.  `prompt`: a prompt chunk
 *  (the prefill GEMM at every width).  Scratch (the schedule, the prefill operand planes) is the MMQ arena
 *  slot's.  Refuses (-1) on any contract violation; nothing is launched past a refusal. */
int exl3_moe_routed_launch(const exl3_moe_stacks &w, const int32_t *sel, const float *wts, int T, int top_k,
                           float clamp, const exl3_moe_act &x, const exl3_moe_bufs &b, float *out,
                           uint32_t *nf_flag, uint32_t nf_code, bool prompt, cudaStream_t stream,
                           const exl3_moe_shared *shared = nullptr);

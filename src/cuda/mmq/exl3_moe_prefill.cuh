// SPDX-License-Identifier: MIT
// The EXL3 routed-expert arm for PROMPT CHUNKS, one arm for every family (L251 Qwen, L287 DeepSeek):
// a grouped fp16 tensor-core GEMM over the expert-sorted schedule (mm_ids_helper's ids_src1 / ids_dst /
// expert_bounds), writing the same UNROTATED z as the trellis GEMVs (ds4_exl3_gemv.cuh), so the folds
// and the sums downstream are the GEMV's.  exl3_moe_prefill.cu has the design.
//
// The activation is read in the format its PRODUCER emitted (rule 3): the Qwen family's row-major
// bf16 rows, or the DeepSeek family's E4M3 slot (per-32 ue8m0 plane in the pulsar_mx_sfoff swizzle).
// The arm decodes either into its one fp16 operand plane; an E4M3 value times its power-of-two scale
// is exact in that plane, so on DeepSeek's down (whose input the fold already rotated) the operand is
// the slot's value bit for bit.

#pragma once

#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

/** The activation formats the PREP reads -- each one a producer's, never a conversion of another. */
enum exl3_moe_act_format {
    EXL3_MOE_ACT_BF16_ROWS = 0,   ///< [rows][K] bf16 (the Qwen family; no E4M3 slot there)
    EXL3_MOE_ACT_E4M3_SLOT = 1,   ///< [rows][K] E4M3 + the ue8m0 plane (the DeepSeek family's MX slot)
};

/** The activation as its producer handed it over.  A row is ids_src[assignment] (the token for gate / up,
 *  the pair for a down whose input is per pair). */
struct exl3_moe_act {
    int         format;   ///< exl3_moe_act_format
    const void *x;        ///< bf16 or E4M3 [rows][K]
    const void *sf;       ///< E4M3: the ue8m0 plane (pulsar_mx_sfoff); bf16: NULL
    int         kbp;      ///< E4M3: the plane's blocks per row (pulsar_mx_kbp); bf16: 0
};

/** Which routed EXL3 arm a call takes -- the ONE rule, for every family: a prompt chunk always takes the
 *  prefill GEMM (so a prompt cut anywhere equals one prefilled whole: a row's bits never depend on its
 *  chunk), and so does a decode-row call of at least EXL3_MOE_PREFILL_MIN_ASSIGN assignments (161 = more
 *  than Qwen's 16 tokens x 10 slots, where the GEMM passed the GEMV, L251).  Everything else takes the GEMV. */
constexpr int64_t EXL3_MOE_PREFILL_MIN_ASSIGN = 161;
__host__ __device__ constexpr inline bool exl3_moe_prefill_takes(bool prompt, int64_t n_assign) {
    return prompt || n_assign >= EXL3_MOE_PREFILL_MIN_ASSIGN;
}

/** Device workspace one launch needs for n_assign assignments of a K-wide input: the fp16 operand plane,
 *  its per-(assignment, 128-block) inverse prescales and the CTA work list. */
size_t exl3_moe_prefill_ws_bytes(int64_t n_assign, int K, int n_experts);

/** out[ids_dst[a]][M] f32 = the UNROTATED z of assignment a's projection through table (exl3_expert_table()'s
 *  [trellis, scales] pairs at rate k2): rotate_input = the projection rotates its own input (x * suh, H128 --
 *  gate / up, fused or not); false = a down whose input arrives rotated.  M % 128, K % 128; k2 one of
 *  exl3_arm_has_rate(EXL3_ARM_MOE_PREFILL).  ws: exl3_moe_prefill_ws_bytes(n_assign, K, n_experts) bytes,
 *  256-aligned.  Agrees with the GEMV to rounding, not to the bit. */
int exl3_moe_prefill_launch(
        const void *table, int k2, bool rotate_input, const exl3_moe_act &act,
        const int32_t *ids_dst, const int32_t *ids_src, const int32_t *expert_bounds,
        float *out, int M, int K, int64_t n_assign, int n_experts,
        void *ws, size_t ws_bytes, cudaStream_t stream);

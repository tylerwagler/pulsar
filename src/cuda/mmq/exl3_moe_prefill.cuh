// SPDX-License-Identifier: MIT
// The EXL3 routed-expert projections, every family, on fp16 tensor cores: the arm for PROMPT CHUNKS (L251
// Qwen, L287 DeepSeek: a grouped GEMM) and the arm for DECODE rows (task #46: a split-K GEMM), over the
// expert-sorted schedule (mm_ids_helper's ids_src1 / ids_dst / expert_bounds), each writing the UNROTATED z
// the fold and the sum (exl3_moe_epilogue.cuh) consume.  The row kind picks the arm (exl3_moe.cu: a prompt
// chunk the grouped GEMM, so a prompt cut anywhere equals one prefilled whole; decode rows the decode GEMM,
// whose split count depends on K alone, so a decode row's bits never depend on how many rows ride with it).
// exl3_moe_prefill.cu has the design.
//
// The activation is read in the format its PRODUCER emitted (rule 3): the Qwen family's row-major
// bf16 rows, or the DeepSeek family's E4M3 slot (per-32 ue8m0 plane in the pulsar_mx_sfoff swizzle).
// Both arms share one PREP into one fp16 operand plane; an E4M3 value times its power-of-two scale
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

/** Device workspace one launch needs for n_assign assignments of a K-wide input: the fp16 operand plane,
 *  its per-(assignment, 128-block) inverse prescales and the CTA work list. */
size_t exl3_moe_prefill_ws_bytes(int64_t n_assign, int K, int n_experts);

/** out[ids_dst[a]][M] f32 = the UNROTATED z of assignment a's projection through table (exl3_expert_table()'s
 *  [trellis, scales] pairs at rate k2): rotate_input = the projection rotates its own input (x * suh, H128 --
 *  gate / up, fused or not); false = a down whose input arrives rotated.  M % 128, K % 128; k2 one of
 *  exl3_arm_has_rate(EXL3_ARM_MOE_PREFILL).  ws: exl3_moe_prefill_ws_bytes(n_assign, K, n_experts) bytes,
 *  256-aligned.  Agrees with the decode arm to rounding, not to the bit. */
int exl3_moe_prefill_launch(
        const void *table, int k2, bool rotate_input, const exl3_moe_act &act,
        const int32_t *ids_dst, const int32_t *ids_src, const int32_t *expert_bounds,
        float *out, int M, int K, int64_t n_assign, int n_experts,
        void *ws, size_t ws_bytes, cudaStream_t stream);

/** The most assignments one decode launch takes (its grid spans them). */
constexpr int64_t EXL3_MOE_DECODE_MAX_ASSIGN = 65535;

/** What a decode launch leaves for its consumer: per projection, `splits` partial planes [rows][M] f32 (row =
 *  ids_dst) `split_stride` floats apart; the consumer adds them in split order.  z[1] is the pair's up. */
struct exl3_moe_partials {
    float  *z[2];
    int     splits;
    int64_t split_stride;
};

/** Device workspace a decode launch of n_proj (1, or 2 for a gate + up pair) projections K -> M over n_assign
 *  assignments needs: per projection the PREP's fp16 plane and prescales and the split partials. */
size_t exl3_moe_decode_ws_bytes(int n_proj, int64_t n_assign, int K, int M);

/** The routed arm for DECODE rows (task #46): the PREP above (the prefill arm's operand, bit for bit), then an
 *  fp16 tensor-core GEMM split over K -- a CTA per (64 outputs, a 16-assignment tile of one expert, a split; the
 *  PREP plans the tiles, so an expert's weights stream once for all of its rows) -- whose split count is a
 *  function of K alone, so a row's bits never depend on how many rows ride with it.
 *  table1 non-NULL: a gate + up pair, both projections in the two launches.  The partials are its z:
 *  *out says where they are.  Agrees with the prefill arm to rounding, not to the bit. */
int exl3_moe_decode_launch(
        const void *table0, const void *table1, int k2, bool rotate_input, const exl3_moe_act &act,
        const int32_t *ids_dst, const int32_t *ids_src, const int32_t *expert_bounds,
        int M, int K, int64_t n_assign, int n_experts, void *ws, size_t ws_bytes,
        exl3_moe_partials *out, cudaStream_t stream);

// SPDX-License-Identifier: MIT
// ds4_act_block.cuh - the MMQ activation staging block, owned by us.
//
// WHAT IT HOLDS.  E4M3 values with one ue8m0 scale per 32, which is what every
// GEMM input in this engine is.  `qs[]` carries e4m3 BIT PATTERNS and `d4[i]`
// carries the ue8m0 byte stored as a float -- see ds4_quantize_e4m3.cu, which
// is the only thing that fills it.
//
// WHY THE LAYOUT LOOKS LIKE THIS.  It began as upstream llama.cpp's
// `block_q8_1_mmq` and kept the layout BYTE FOR BYTE when the routed-expert
// path moved from int8 q8_1 to E4M3: that struct already meant "1 scale per 32
// values", so reusing it left load_B_tile's ldmatrix, the prefetch pipeline and
// every stride derivation untouched.  It costs 12 bytes per 144-byte block to
// carry a one-byte exponent in a float slot, and not touching a hand-tuned
// staging pipeline is worth more than 8% of a staging buffer.
//
// WHY IT LIVES HERE.  It was renamed from `block_q8_1_mmq` on 2026-08-18 because
// the old name was read as evidence that the MoE still ran q8_1 activations, and
// a whole scoping verdict got written on that mistake (ledger L065).  The
// DEFINITION moves out of mmq.cuh for the reason in L066: nothing calls mmq.cuh's
// kernels any more -- the live IQ2 path is our own D2R MXFP8 GEMM -- so the
// staging struct was the last thing tying live code to a 1630-line vendored
// header that is on the deletion list.
//
// The size assert is a LITERAL 144 rather than `4*sizeof(block_q8_1)`, which is
// how upstream wrote it.  144 is the contract every stride in the staging path
// is derived from; expressing it through a ggml type would re-create the
// dependency this header exists to remove.

#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

// One MX scale block is 32 values; a staging block carries four of them.
static constexpr int DS4_ACT_BLOCK_QK   = 32;
static constexpr int DS4_ACT_BLOCK_VALS = 4 * DS4_ACT_BLOCK_QK;   // 128

struct block_mx_act_mmq {
    // The scale slot.  Only `d4` is used by the E4M3 producer (one ue8m0 byte
    // per 32 values, widened to float); the other two members are upstream's
    // q8_1 interpretations, kept so the union's size and alignment are
    // unchanged and so any surviving vendored code still type-checks.
    union {
        float d4[4];    // 1 scale per 32 values -- ue8m0 byte as float (E4M3 path)
        half2 ds4[4];   // upstream q8_1: 16-bit scale + 16-bit partial sum per 32
        half  d2s6[8];  // upstream q8_1: scale per 64 + partial sums
    };
    int8_t qs[DS4_ACT_BLOCK_VALS];   // e4m3 bit patterns (NOT int8 magnitudes)
};

static_assert(sizeof(block_mx_act_mmq) == 144,
              "staging block must stay 144 B: every stride in the MMQ staging "
              "path is derived from this size");

// Launch geometry for the staging kernels.  This was upstream's
// CUDA_QUANTIZE_BLOCK_SIZE_MMQ in quantize.cuh; it is the only thing the live
// E4M3 staging still needed from that header, and keeping a vendored file alive
// to supply one integer is the wrong trade.
static constexpr int DS4_ACT_QUANT_BLOCK = 128;

// Batch-size knee used by ds4_mmq_should_use.  Upstream's
// MMQ_DP4A_MAX_BATCH_SIZE from mmq.cuh: "max. batch size to use for dp4a MMQ
// kernels when FP16 tensor cores are available".  The dp4a kernels it referred
// to are deleted (L066); the CONSTANT survives because the gate that decides
// whether the caller takes the MMQ route at all still reads it, and that gate is
// live.  Kept at upstream's value so the routing decision is unchanged by the
// deletion -- this extraction is not the place to retune a threshold.
#define MMQ_DP4A_MAX_BATCH_SIZE 64

/** @file
 *  The GQA attention block's one implementation: prep, attention, combine, commit (L295 F1). */
/* GQA attention block (L295 F1).  The contract, the KV row layout and the
 * arithmetic live in pulsar_cuda_attn_gqa.h; this file is the one
 * implementation.  Kernels, in launch order, per call:
 *
 *   gqa_prep      per row: q heads (norm, temperature) -> bf16 workspace; K
 *                 heads (norm) and V heads -> the call's fresh KV rows
 *   gqa_attn      a tile of up to 64 / G consecutive rows of one sequence x
 *                 the G query heads of KV head g = 64 MMA rows (4 warps x
 *                 m16, Q held as A fragments in registers).  Per 64-key split: the relative bias (r . P on the
 *                 tensor cores), S = Q K^T, the online softmax, O += P V.
 *                 Chunk = 4 splits.  <FOLD = false>: one CTA per (chunk, KV
 *                 head, tile), the chunk's (m, l, acc) to the workspace;
 *                 <FOLD = true>: one CTA per (KV head, tile) folds its chunks
 *                 in registers and runs the epilogue.
 *   gqa_combine   (split schedule) per (row, query head): fold the row's
 *                 chunks in order, the epilogue
 *   gqa_commit    the call's fresh KV rows into the pools (a ring keeps only
 *                 its last cap positions)
 *
 * Keys of THIS call come from the fresh rows, earlier keys from the pool, so
 * the attention never reads a slot the same call overwrites. */
#include "pulsar_cuda_attn_gqa.h"
#include "pulsar_cuda_internal.h"
#include "pulsar_cuda_rope.cuh"   /* the NeoX bf16 rotation: ONE authority (rope_neox_bf16_cs_dev / _rotate_dev) */
#include "pulsar_cuda_arch.cuh"

#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <vector>

namespace {

/// Keys per split: one 64-key QK / PV tile of the attention kernel.
constexpr uint32_t GQA_SPLIT        = 64u;    /* keys per split: one QK / PV tile */
/// Splits per chunk, the unit one partial (m, l, acc) covers.
constexpr uint32_t GQA_CHUNK_SPLITS = 4u;
/// Keys per chunk (256), cut at absolute positions: chunk c holds keys [256 c, 256 c + 256).
constexpr uint32_t GQA_CHUNK        = GQA_SPLIT * GQA_CHUNK_SPLITS;   /* keys per partial */
/// Warps per attention CTA (128 threads), each owning 16 MMA rows.
constexpr uint32_t GQA_WARPS        = 4u;
/// MMA rows per attention CTA: (row, query head) pairs = tile rows x G.
constexpr uint32_t GQA_MROWS        = 16u * GQA_WARPS;   /* (row, head) pairs per CTA */
/// Row count at and above which a call takes the folded schedule (no partials, no combine).
constexpr uint32_t GQA_FOLD_ROWS    = 64u;    /* calls at least this wide fold in the CTA */
/// Bias distances per warp tile: 64 keys + 15 further rows = 79, rounded up to 10 n8 tiles.
constexpr uint32_t GQA_BIAS_W       = 80u;    /* distances a warp's bias tile spans: 64 keys + 15 rows */
/// Widest query group G = n_q / n_kv: one CTA's MMA rows.
constexpr uint32_t GQA_MAX_GROUP    = GQA_MROWS;
/// The only built relative-bias r width per head: one m16n8k16 k-step.
constexpr uint32_t GQA_REL_DIM      = 16u;   /* r width per head: one MMA k-step */
/// log2(e): logits are scaled into base-2 units so the online softmax runs on exp2f.
constexpr float    GQA_LOG2E        = 1.4426950408889634f;

static_assert(GQA_SPLIT == 64u, "the warp layout tiles 64 keys: 8 QK n-tiles, 4 PV k-steps");

/* One row of the call. */
/// One row's device descriptor (48 B), filled by pulsar_gqa_plan; positions are absolute token positions.
struct gqa_row {
    /// Device base of the sequence's KV pool (slot = pos, or pos % cap on a ring); NULL for a poolless block run.
    uint8_t *cache;     /* the sequence's pool */
    /// Absolute position of the row's token.
    uint32_t pos;       /* this row's position */
    /// First key position attended (0, or pos + 1 - window on a WINDOW span).
    uint32_t lo;        /* the first key it attends */
    /// Position of the sequence's first row in this call; keys >= p0 come from the fresh rows.
    uint32_t p0;        /* the sequence's first position in this call */
    /// Call row index holding p0 (key x's fresh row is fresh0 + x - p0).
    uint32_t fresh0;    /* the call row holding p0 */
    /// Position of the sequence's last row in this call.
    uint32_t p_last;    /* the sequence's last position in this call */
    /// Pool slots: FULL token capacity, WINDOW ring size.
    uint32_t cap;       /* pool slots */
    /// 1 on a WINDOW span (slot = pos % cap), else 0.
    uint32_t ring;      /* 1: slot = position % cap */
    /// The row's first chunk index, lo / GQA_CHUNK.
    uint32_t chunk0;    /* lo / GQA_CHUNK */
    /// Log-length temperature at pos (gqa_tau), applied to the stored q.
    float    tau;       /* the log-length temperature (1 when the layer declares none) */
    /// Last key position attended (inclusive).
    uint32_t hi;        /* the last key it attends: its own position, or (a block) its run's last */
};
static_assert(sizeof(gqa_row) == 48, "row descriptor");

/* Up to 64 / G consecutive rows of one sequence. */
/// A query tile: rows row0 .. row0 + n - 1, consecutive rows of one sequence, n <= GQA_MROWS / G.
struct gqa_tile { uint32_t row0, n; };
/** @var gqa_tile::row0
 *  First call row of the tile.
 *  @var gqa_tile::n
 *  Rows in the tile. */

/// The attention and combine kernels' argument block (device pointers; strides in elements).
struct gqa_args {
    /// Row descriptors, [n_rows].
    const gqa_row        *rows;
    /// Tile descriptors, [n_tiles]; blockIdx.z indexes them.
    const gqa_tile       *tiles;
    /// The prepped q (normed, roped, times tau), bf16.
    const __nv_bfloat16  *q;        /* workspace [rows][n_q][HD] */
    /// The call's own KV rows in the layer's row format; keys at positions >= p0 are read here.
    const uint8_t        *fresh;    /* workspace [rows][token_bytes] */
    /// Relative-bias r input, element (row, h, i) at r[row * ld_r + h * r_hs + i]; NULL without a bias.
    const __nv_bfloat16  *r;
    /// r row stride, elements.
    uint64_t              ld_r;
    /// r head stride, elements.
    uint32_t              r_hs;
    /// The layer's relative-bias table P, bf16.
    const __nv_bfloat16  *proj;     /* [rel_dim][extent] */
    /// Bias distances 0 .. extent - 1 (0 beyond); 0 = no bias.
    uint32_t              extent;
    /// Softmax scale on q . k.
    float                 scale;
    /// Query heads, KV heads, group G = n_q / n_kv, KV row bytes per token, and the partials' chunk stride.
    uint32_t              n_q, n_kv, G, token_bytes, max_chunks;
    /// Split schedule: per-chunk unnormalised accumulators, f32 (unused when folding).
    float                *part;     /* split schedule: [rows][n_q][max_chunks][HD] */
    /// Split schedule: per-chunk (running max in log2 units, softmax denominator).
    float2               *ml;       /*                 [rows][n_q][max_chunks] */
    /// Output gate input (y *= sigmoid(gate)), element (row, h, d) at gate[row * ld_gate + h * gate_hs + d]; NULL = none.
    const __nv_bfloat16  *gate;
    /// gate row stride, elements.
    uint64_t              ld_gate;
    /// gate head stride, elements.
    uint32_t              gate_hs;
    /// Output bf16, head h of row r at out[r * ld_out + h * HD ..].
    __nv_bfloat16        *out;
    /// out row stride, elements.
    uint64_t              ld_out;
    /// Optional f32 copy of the output, [rows][n_q * HD]; NULL in the lane.
    float                *tap;
    /// The step's non-finite flag (pulsar_gqa_io::nf_flag, or pulsar_gqa_forward's own word): where the arms widen
    /// (sm_75), an operand that is not finite as fp16 CASes nf_code into it (gqa_nf_raise).  Unconditional members:
    /// the host pass builds this block and must lay it out as every device pass reads it; sm_80+ never reads them.
    uint32_t             *nf;
    /// The code recorded there, nonzero where the arms widen.
    uint32_t              nf_code;
};

/** @var gqa_args::n_kv
 *  KV heads.
 *  @var gqa_args::G
 *  Query group width n_q / n_kv.
 *  @var gqa_args::token_bytes
 *  KV row bytes per token.
 *  @var gqa_args::max_chunks
 *  Chunk stride of part / ml (gqa_max_chunks at the call's context). */

#if !PULSAR_SM80
/* A packed bf16 pair as the packed fp16 pair of the SAME values: a bf16 has 8 significant bits and fp16 11, so the
 * widening is exact for every value inside fp16's normal range (|x| in [2^-14, 65504]).  The attention's operands sit
 * far inside it (q / k / v peak ~32 in the Nemotron-3.5 encoders, retriever/README.md; P and its lo part are in
 * [0, 1]), and a P lo term small enough to land in fp16's subnormals carries at most 2^-8 of a weight already < 2^-14.
 * ABOVE the range the widening is not the value: a finite bf16 past 65504 rounds to an fp16 infinity.  That is checked
 * where a q / k / v / r / P operand is widened (gqa_opc) and refuses the step (gqa_nf_raise; O59, L375). */
/// @param w a packed bf16x2 word.
/// @return the same two values as a packed fp16x2 word (sm_75 only).
__device__ __forceinline__ uint32_t gqa_bf2h(uint32_t w) {
    const __half2 h = __floats2half2_rn(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u));
    return *reinterpret_cast<const uint32_t *>(&h);
}
#endif

/// The MMA's operand form of a packed bf16x2 word (L349): itself on sm_80, its exact fp16 widening (gqa_bf2h) on
/// sm_75 -- so a kernel widens a fragment ONCE where it is loaded and every MMA that reuses it (a K fragment across
/// a verify's rows, a V fragment across rows and the P hi / lo pair) takes it as is.  Same values, same MMAs: the
/// bytes of a kernel that widens once are the bytes of one that widens at every MMA.
/// @param w a packed bf16x2 word.
/// @return the word in the MMA's operand form.
__device__ __forceinline__ uint32_t gqa_op(uint32_t w) {
#if PULSAR_SM80
    return w;
#else
    return gqa_bf2h(w);
#endif
}
/// gqa_op of a q / k / v / r / P operand word, CHECKED (O59): on sm_75 `bad` gathers whether either widened half is an
/// fp16 infinity or NaN -- a finite bf16 above 65504 in magnitude, or an operand that was not finite to begin with --
/// which the kernel then reports (gqa_nf_raise).  The widened word is gqa_op's, bit for bit.  On sm_80 it is gqa_op:
/// the bf16 MMA carries every bf16 value exactly and there is nothing to check.
/// @param w a packed bf16x2 word.
/// @param bad accumulates nonzero when a half left fp16's range (sm_75; untouched on sm_80).
/// @return the word in the MMA's operand form.
__device__ __forceinline__ uint32_t gqa_opc(uint32_t w, uint32_t &bad) {
#if PULSAR_SM80
    (void)bad;
    return w;
#else
    const uint32_t h = gqa_bf2h(w);
    /* a half's exponent field (0x7c00) all ones: adding 0x0400 carries into its bit 15, and only then */
    bad |= ((h & 0x7c007c00u) + 0x04000400u) & 0x80008000u;
    return h;
#endif
}
/// gqa_opc on the four words of a 16-byte load (a K / V row's 8 elements as staged to smem).
/// @param v four packed bf16x2 words.
/// @param bad accumulates nonzero when an element left fp16's range (sm_75).
/// @return the same in the MMA's operand form.
__device__ __forceinline__ uint4 gqa_opc4(uint4 v, uint32_t &bad) {
    return make_uint4(gqa_opc(v.x, bad), gqa_opc(v.y, bad), gqa_opc(v.z, bad), gqa_opc(v.w, bad));
}
/// Reports what gqa_opc gathered: on sm_75 a nonzero `bad` CASes the call's code into the step's non-finite flag
/// (first writer wins; the caller reads it at the step's end and refuses the step).  Nothing on sm_80.
/// @param a the kernel's arguments (the flag and the code).
/// @param bad gqa_opc's accumulator.
__device__ __forceinline__ void gqa_nf_raise(const gqa_args &a, uint32_t bad) {
#if PULSAR_SM80
    (void)a;
    (void)bad;
#else
    if (bad) atomicCAS(a.nf, 0u, a.nf_code);
#endif
}
/// One m16n8k16 MMA with f32 accumulate, d += A B, over operands already in gqa_op form (on sm_75 two fp16 m16n8k8
/// MMAs, k 0..7 then k 8..15).
/// @param d the accumulator fragment, updated in place.
/// @param a0 A fragment register 0 (gqa_op form).
/// @param a1 A fragment register 1.
/// @param a2 A fragment register 2.
/// @param a3 A fragment register 3.
/// @param b0 B fragment register 0 (gqa_op form).
/// @param b1 B fragment register 1.
__device__ __forceinline__ void gqa_mma_op(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                           uint32_t b0, uint32_t b1) {
#if PULSAR_SM80
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    /* sm_75 (L321): Turing has no bf16 MMA and no k16 shape.  The same products as fp16 operands on two m16n8k8 MMAs:
     * the k16 fragment's registers split along k exactly so (A a0 a1 / B b0 hold k 0..7, a2 a3 / b1 hold k 8..15),
     * the split exllamav3's sm_75 port uses -- pulsar_mma_f16_16816_k8, the one definition (pulsar_cuda_arch.cuh).
     * f32 accumulate as on sm_80. */
    const uint32_t A[4] = {a0, a1, a2, a3};
    pulsar_mma_f16_16816_k8(d, A, b0, b1);
#endif
}
/// One m16n8k16 bf16 MMA with f32 accumulate, d += A B (on sm_75 two fp16 m16n8k8 MMAs, the operands widened and
/// checked: gqa_opc).
/// @param d the accumulator fragment, updated in place.
/// @param a0 A fragment register 0.
/// @param a1 A fragment register 1.
/// @param a2 A fragment register 2.
/// @param a3 A fragment register 3.
/// @param b0 B fragment register 0.
/// @param b1 B fragment register 1.
/// @param bad gqa_opc's accumulator.
__device__ __forceinline__ void gqa_mma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1, uint32_t &bad) {
    gqa_mma_op(d, gqa_opc(a0, bad), gqa_opc(a1, bad), gqa_opc(a2, bad), gqa_opc(a3, bad), gqa_opc(b0, bad),
               gqa_opc(b1, bad));
}
/// ldmatrix .x4 of four 8x8 b16 tiles from shared memory.
/// @param r the four fragment registers.
/// @param p this lane's 16-byte-aligned shared row address.
__device__ __forceinline__ void gqa_ldsm4(uint32_t (&r)[4], const void *p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
/// ldmatrix .x4 .trans: as gqa_ldsm4, each 8x8 tile transposed.
/// @param r the four fragment registers.
/// @param p this lane's 16-byte-aligned shared row address.
__device__ __forceinline__ void gqa_ldsm4t(uint32_t (&r)[4], const void *p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
/* bf16 bits -> f32 is a left shift */
/// @param b bf16 bits.
/// @return the value as f32.
__device__ __forceinline__ float gqa_bf(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
/// @param x an f32 value.
/// @return x rounded to bf16 (round to nearest even), as f32.
__device__ __forceinline__ float gqa_round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
/* f32 pair -> packed bf16x2 (first element in the low half) */
/// @param x0 first value (low half).
/// @param x1 second value (high half).
/// @return the bf16x2 word, each rounded to nearest even.
__device__ __forceinline__ uint32_t gqa_pack(float x0, float x1) {
    const __nv_bfloat162 h = __floats2bfloat162_rn(x0, x1);
    return *reinterpret_cast<const uint32_t *>(&h);
}
/* f32 pair -> (hi, lo) bf16x2 words: hi = bf16(x), lo = bf16(x - hi) */
/// @param x0 first value.
/// @param x1 second value.
/// @param hi out: bf16x2 of the rounded values.
/// @param lo out: bf16x2 of the residuals x - hi.
__device__ __forceinline__ void gqa_split2(float x0, float x1, uint32_t &hi, uint32_t &lo) {
    const float h0 = gqa_round(x0), h1 = gqa_round(x1);
    hi = gqa_pack(h0, h1);
    lo = gqa_pack(__fsub_rn(x0, h0), __fsub_rn(x1, h1));
}

/* THE recurrence: fold a reduced block (m, l, a) into a running (M, L, A). */
/// Online-softmax merge in base 2: M = max(M, m); L and A rescaled by exp2 of the max shifts, then summed.
/// @param M running max (log2 units), in/out.
/// @param L running denominator, in/out.
/// @param A running unnormalised accumulator, in/out.
/// @param m the block's max.
/// @param l the block's denominator.
/// @param a the block's accumulator.
__device__ __forceinline__ void gqa_fold(float &M, float &L, float &A, float m, float l, float a) {
    const float mn = fmaxf(M, m);
    const float c1 = exp2f(__fsub_rn(M, mn)), c2 = exp2f(__fsub_rn(m, mn));
    L = __fadd_rn(__fmul_rn(L, c1), __fmul_rn(l, c2));
    A = __fadd_rn(__fmul_rn(A, c1), __fmul_rn(a, c2));
    M = mn;
}
/* THE epilogue: normalise, times the optional sigmoid gate. */
/// @param A the folded accumulator.
/// @param L the folded denominator.
/// @param gate the (row, head, dim) gate element, or NULL for no gate.
/// @return A / L, times sigmoid(gate) when gated.
__device__ __forceinline__ float gqa_epilogue(float A, float L, const __nv_bfloat16 *gate) {
    const float y = __fdiv_rn(A, L);
    if (!gate) return y;
    const float g = __bfloat162float(*gate);
    return __fmul_rn(y, __fdiv_rn(1.0f, __fadd_rn(1.0f, __expf(-g))));
}

/// Butterfly sum over a full warp in a fixed order (every lane gets the same bits).
/// @param v this lane's value.
/// @return the warp's sum.
__device__ __forceinline__ float gqa_warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

/* ---- prep: one warp per head vector ------------------------------------------ */
/// The prep kernel's argument block (device pointers; strides in elements).
struct gqa_prep_args {
    /// Row descriptors (pos and tau are read).
    const gqa_row        *rows;
    /// The q / k / v projection outputs, bf16 (pulsar_gqa_in.ptr).
    const __nv_bfloat16  *qin, *kin, *vin;
    /// Row strides of qin / kin / vin, elements.
    uint64_t              ld_q, ld_k, ld_v;
    /// Head strides of qin / kin / vin, elements.
    uint32_t              hs_q, hs_k, hs_v;
    /// q and k RMSNorm weights, bf16 [HD].
    const uint16_t       *qw, *kw;    /* NULL: no norm */
    /// RMSNorm epsilon.
    float                 eps;
    /// Query heads, KV heads, KV row bytes per token.
    uint32_t              n_q, n_kv, token_bytes;
    /// Rope base; 0 = no rope.
    float                 rope_theta; /* > 0: the NeoX rotation of q and k at the row's position */
    /// KV row element format flag.
    uint32_t              e4m3;       /* 1: the KV rows are E4M3 bytes (PULSAR_GQA_KV_E4M3), else bf16 */
    /// Out: prepped q, bf16 [rows][n_q][HD].
    __nv_bfloat16        *q;          /* workspace */
    /// Out: the call's KV rows, [rows][token_bytes].
    uint8_t              *fresh;      /* workspace */
};
/** @var gqa_prep_args::kin
 *  The k projection output, bf16.
 *  @var gqa_prep_args::vin
 *  The v projection output, bf16.
 *  @var gqa_prep_args::ld_k
 *  Row stride, elements.
 *  @var gqa_prep_args::ld_v
 *  Row stride, elements.
 *  @var gqa_prep_args::hs_k
 *  Head stride, elements.
 *  @var gqa_prep_args::hs_v
 *  Head stride, elements.
 *  @var gqa_prep_args::kw
 *  k RMSNorm weight, bf16 [HD].
 *  @var gqa_prep_args::n_kv
 *  KV heads.
 *  @var gqa_prep_args::token_bytes
 *  KV row bytes per token. */

/* L321: a scaled rope's (YaRN) per-pair inverse frequencies ride a DERIVED argument block, so the plain kernel's
 * parameters -- and its code -- are exactly what they were */
/// gqa_prep_args plus a scaled rope's per-pair inverse frequency table.
struct gqa_prep_table_args : gqa_prep_args {
    /// Device f32 [HD / 2]: pair p's inverse frequency (pulsar_gqa_layer::rope_inv_freq).
    const float *rope_inv;
};

/* A = gqa_prep_args (the plain rope: 1 / theta^(2p/d)) or gqa_prep_table_args (the frequencies from a.rope_inv).
 * A template, not a branch: the plain instantiation is the kernel it was. */
/// Per row, one warp per head vector: q heads (norm, rope, times tau) into a.q, K heads (norm, rope) and V heads into
/// the fresh KV rows in the row format.  Grid (n_rows, ceil((n_q + 2 n_kv) / 8)), 256 threads (8 warps = 8 jobs).
/// L338: the CTA's row's bf16 cos / sin per pair first, into shared memory (one double cos / sin per thread at most),
/// then every head rotates from them -- the same values the per-element evaluation gave.
/// @tparam HD head_dim (64, 128, 256 or 512); lane l holds elements l + 32 j.
/// @tparam A gqa_prep_args or gqa_prep_table_args.
/// @param a the arguments.
template <int HD, class A>
__global__ void __launch_bounds__(256) gqa_prep_kernel(const A a) {
    constexpr bool TABLE = std::is_same<A, gqa_prep_table_args>::value;
    constexpr int NJ = HD / 32;
    const uint32_t row = blockIdx.x;
    const uint32_t job = blockIdx.y * 8u + (threadIdx.x >> 5);
    const uint32_t lane = threadIdx.x & 31u;
    __shared__ float s_cos[HD / 2], s_sin[HD / 2];
    const bool rope = a.rope_theta > 0.f;
    if (rope) {
        const uint32_t pos = a.rows[row].pos;
        for (uint32_t pr = threadIdx.x; pr < HD / 2; pr += blockDim.x) {
            float inv;
            if constexpr (TABLE) inv = a.rope_inv[pr];
            else inv = rope_neox_inv_freq_dev(pr, HD, a.rope_theta);
            rope_neox_bf16_cs_dev(inv, pos, &s_cos[pr], &s_sin[pr]);
        }
        __syncthreads();
    }
    if (job >= a.n_q + 2u * a.n_kv) return;
    const __nv_bfloat16 *src;
    const uint16_t *w = nullptr;
    __nv_bfloat16 *dst;
    uint8_t *dst8 = nullptr;   /* an E4M3 KV row's head */
    float tau = 1.0f;
    const uint32_t esz = a.e4m3 ? 1u : 2u;
    if (job < a.n_q) {
        src = a.qin + row * a.ld_q + (uint64_t)job * a.hs_q;
        w = a.qw;
        tau = a.rows[row].tau;
        dst = a.q + ((uint64_t)row * a.n_q + job) * HD;
    } else if (job < a.n_q + a.n_kv) {
        const uint32_t g = job - a.n_q;
        src = a.kin + row * a.ld_k + (uint64_t)g * a.hs_k;
        w = a.kw;
        dst8 = a.fresh + (uint64_t)row * a.token_bytes + (uint64_t)g * HD * esz;
        dst = reinterpret_cast<__nv_bfloat16 *>(dst8);
    } else {
        const uint32_t g = job - a.n_q - a.n_kv;
        src = a.vin + row * a.ld_v + (uint64_t)g * a.hs_v;
        dst8 = a.fresh + (uint64_t)row * a.token_bytes + (uint64_t)(a.n_kv + g) * HD * esz;
        dst = reinterpret_cast<__nv_bfloat16 *>(dst8);
    }
    if (!a.e4m3) dst8 = nullptr;
    float x[NJ];
    #pragma unroll
    for (int j = 0; j < NJ; j++) x[j] = __bfloat162float(src[lane + 32 * j]);
    if (w) {
        float ss = 0.f;
        #pragma unroll
        for (int j = 0; j < NJ; j++) ss = __fmaf_rn(x[j], x[j], ss);
        ss = gqa_warp_sum(ss);
        const float rstd = rsqrtf(__fadd_rn(__fdiv_rn(ss, (float)HD), a.eps));
        #pragma unroll
        for (int j = 0; j < NJ; j++) x[j] = gqa_round(__fmul_rn(__fmul_rn(x[j], rstd), gqa_bf(w[lane + 32 * j])));
    }
    /* L298: the rope of q and k (never v) after the norm -- element lane + 32 j pairs with lane + 32 (j + NJ / 2) */
    if (rope && job < a.n_q + a.n_kv) {
        #pragma unroll
        for (int j = 0; j < NJ / 2; j++) {
            float r0, r1;
            const uint32_t pr = lane + 32u * (uint32_t)j;
            rope_neox_bf16_rotate_dev(x[j], x[j + NJ / 2], s_cos[pr], s_sin[pr], &r0, &r1);
            x[j] = r0;
            x[j + NJ / 2] = r1;
        }
    }
    /* L298: an E4M3 KV row stores each (bf16-valued) element as one satfinite E4M3 byte at unit scale */
    if (dst8) {
        #pragma unroll
        for (int j = 0; j < NJ; j++) dst8[lane + 32 * j] = (uint8_t)__nv_cvt_float_to_fp8(x[j], __NV_SATFINITE, __NV_E4M3);
        return;
    }
    /* the temperature multiplies the stored bf16 q (tau = 1 is an exact identity) */
    #pragma unroll
    for (int j = 0; j < NJ; j++) dst[lane + 32 * j] = __float2bfloat16_rn(__fmul_rn(x[j], tau));
}

/* 8 E4M3 bytes -> 8 bf16 (exact: an E4M3 value is a half and a bf16 value) as one 16-byte word */
/// @param v eight E4M3 bytes, element 0 in the low byte of v.x.
/// @return the eight values as bf16, packed in order.
__device__ __forceinline__ uint4 gqa_e4m3x8_to_bf16(uint2 v) {
    const uint32_t w[2] = {v.x, v.y};
    uint32_t o[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(w[i >> 1] >> (16 * (i & 1))), __NV_E4M3);
        __half2 hh;
        memcpy(&hh, &h, sizeof hh);
        const float2 f = __half22float2(hh);
        o[i] = gqa_pack(f.x, f.y);
    }
    return make_uint4(o[0], o[1], o[2], o[3]);
}

/* ---- attention ------------------------------------------------------------------ */
/// The attention over one tile x KV head g: 64 MMA rows (tile rows x G heads; 4 warps x m16), keys in 64-key splits
/// staged in static shared memory (K and V, pitch HD + 8 bf16; plus the bias tile when BIAS).  Grid (FOLD ? 1 :
/// grid_chunks, n_kv, n_tiles), 128 threads.  FOLD: the CTA walks its tile's chunks, folds in registers and writes
/// out / tap; else CTA x reduces chunk chunk0 + x and writes its (m, l) and accumulator to ml / part.
/// @tparam HD head_dim (64 or 128).
/// @tparam BIAS the relative bias is applied.
/// @tparam FOLD the folded schedule.
/// @tparam E4 the KV rows are E4M3 (widened to bf16 in the tile loader).
/// @param a the arguments.
template <int HD, bool BIAS, bool FOLD, bool E4>
__global__ void __launch_bounds__(128) gqa_attn_kernel(const gqa_args a) {
    constexpr int P = HD + 8;      /* bf16 smem pitch: 16-byte rows, conflict-free ldmatrix */
    constexpr int ND = HD / 8;     /* PV n-tiles */
    constexpr int KS = HD / 16;    /* QK k-steps */
    constexpr int V8 = HD / 8;     /* uint4 per head row */
    __shared__ __align__(16) __nv_bfloat16 ks[GQA_SPLIT * P];
    __shared__ __align__(16) __nv_bfloat16 vs[GQA_SPLIT * P];
    __shared__ __nv_bfloat16 bs[BIAS ? GQA_WARPS * 16u * GQA_BIAS_W : 1u];

    const gqa_tile T = a.tiles[blockIdx.z];
    const uint32_t g = blockIdx.y;
    const gqa_row R0 = a.rows[T.row0];
    const gqa_row RL = a.rows[T.row0 + T.n - 1u];
    const uint32_t c_last = RL.hi / GQA_CHUNK;
    const uint32_t c_first = FOLD ? R0.chunk0 : R0.chunk0 + blockIdx.x;
    if (c_first > c_last) return;
    const uint32_t c_end = FOLD ? c_last : c_first;
    const uint32_t tid = threadIdx.x, w = tid >> 5, lane = tid & 31u, gr = lane >> 2, c = lane & 3u;
    const uint32_t G = a.G, mrows = T.n * G;

    /* this lane's two MMA rows: gr and gr + 8 of the warp's 16 */
    bool mval[2];
    uint32_t mpos[2], mhi[2], mlo[2], mrow[2], mhead[2], mch0[2];
    float mtau[2];
    #pragma unroll
    for (int hh = 0; hh < 2; hh++) {
        const uint32_t m = 16u * w + gr + 8u * hh;
        mval[hh] = m < mrows;
        const uint32_t ri = mval[hh] ? m / G : 0u;
        const gqa_row &Rm = a.rows[T.row0 + ri];
        mpos[hh] = Rm.pos; mhi[hh] = Rm.hi; mlo[hh] = Rm.lo; mtau[hh] = Rm.tau; mch0[hh] = Rm.chunk0;
        mrow[hh] = T.row0 + ri;
        mhead[hh] = g * G + (mval[hh] ? m % G : 0u);
    }
    /* the warp's Q as MMA A fragments, held for the whole call: a0 / a2 row gr, a1 / a3 row gr + 8 */
    uint32_t qa[KS][4];
    uint32_t bad = 0u;   /* O59: a widened q / k / v / r / P operand outside fp16's range (sm_75) */
    #pragma unroll
    for (int hh = 0; hh < 2; hh++) {
        const __nv_bfloat16 *qp = a.q + ((uint64_t)mrow[hh] * a.n_q + mhead[hh]) * HD;
        #pragma unroll
        for (int kk = 0; kk < KS; kk++) {
            qa[kk][hh]     = gqa_opc(mval[hh] ? *reinterpret_cast<const uint32_t *>(qp + kk * 16 + 2u * c) : 0u, bad);
            qa[kk][2 + hh] = gqa_opc(mval[hh] ? *reinterpret_cast<const uint32_t *>(qp + kk * 16 + 8u + 2u * c) : 0u, bad);
        }
    }
    gqa_nf_raise(a, bad);
    const bool warp_live = 16u * w < mrows;
    const uint32_t wpos0 = a.rows[T.row0 + (warp_live ? (16u * w) / G : 0u)].pos;   /* the warp's first row */

    constexpr uint32_t ESZ = E4 ? 1u : 2u;   /* a KV element's bytes (E4M3 rows widen to bf16 in the loader) */
    const uint32_t koff = g * HD * ESZ, voff = (a.n_kv + g) * HD * ESZ;
    float RM[2] = {-INFINITY, -INFINITY}, RLs[2] = {0.f, 0.f};
    float RA[FOLD ? ND : 1][4];
    #pragma unroll
    for (int nd = 0; nd < (FOLD ? ND : 1); nd++) RA[nd][0] = RA[nd][1] = RA[nd][2] = RA[nd][3] = 0.f;

    for (uint32_t ch = c_first; ch <= c_end; ch++) {
        float M[2] = {-INFINITY, -INFINITY}, L[2] = {0.f, 0.f};
        float A[ND][4];
        #pragma unroll
        for (int nd = 0; nd < ND; nd++) A[nd][0] = A[nd][1] = A[nd][2] = A[nd][3] = 0.f;

        for (uint32_t s = 0; s < GQA_CHUNK_SPLITS; s++) {
            const uint32_t x0 = ch * GQA_CHUNK + s * GQA_SPLIT;
            if (x0 > RL.hi) break;
            if (x0 + GQA_SPLIT <= R0.lo) continue;
            __syncthreads();                       /* the previous split is done with ks / vs / bs */
            /* the K tile, then the V tile: each thread's LPT 16-byte loads all issued before its stores */
            #pragma unroll
            for (uint32_t kvh = 0; kvh < 2u; kvh++) {
                constexpr uint32_t LPT = GQA_SPLIT * V8 / 128u;
                const uint32_t off = kvh ? voff : koff;
                __nv_bfloat16 *dst = kvh ? vs : ks;
                uint4 ld[LPT];
                #pragma unroll
                for (uint32_t it = 0; it < LPT; it++) {
                    const uint32_t i = tid + 128u * it, t = i / V8, part = i % V8, x = x0 + t;
                    ld[it] = make_uint4(0u, 0u, 0u, 0u);
                    if (x >= R0.lo && x <= RL.hi) {
                        const uint8_t *rec = x >= R0.p0
                                ? a.fresh + (uint64_t)(R0.fresh0 + (x - R0.p0)) * a.token_bytes
                                : R0.cache + (uint64_t)(R0.ring ? x % R0.cap : x) * a.token_bytes;
                        if constexpr (E4) ld[it] = gqa_e4m3x8_to_bf16(*reinterpret_cast<const uint2 *>(rec + off + part * 8u));
                        else              ld[it] = *reinterpret_cast<const uint4 *>(rec + off + part * 16u);
                    }
                }
                #pragma unroll
                for (uint32_t it = 0; it < LPT; it++) {
                    const uint32_t i = tid + 128u * it, t = i / V8, part = i % V8;
                    *reinterpret_cast<uint4 *>(&dst[t * P + part * 8u]) = gqa_opc4(ld[it], bad);   /* operand form */
                }
            }
            gqa_nf_raise(a, bad);
            __syncthreads();
            if (!warp_live) continue;

            /* ---- the relative bias of this warp's rows: r . P on the tensor cores.  Column
             * n of the tile is distance dbase + n; output (m, n) is one MMA element whatever
             * tile it sits in. */
            const int dbase = (int)wpos0 - (int)x0 - (int)(GQA_SPLIT - 1u);
            const bool bias_live = BIAS && dbase < (int)a.extent;
            if (bias_live) {
                uint32_t ra[4];
                #pragma unroll
                for (int hh = 0; hh < 2; hh++) {
                    const __nv_bfloat16 *rp = a.r + mrow[hh] * a.ld_r + (uint64_t)mhead[hh] * a.r_hs;
                    ra[hh]     = mval[hh] ? *reinterpret_cast<const uint32_t *>(rp + 2u * c) : 0u;
                    ra[2 + hh] = mval[hh] ? *reinterpret_cast<const uint32_t *>(rp + 8u + 2u * c) : 0u;
                }
                const uint16_t *pr = reinterpret_cast<const uint16_t *>(a.proj);
                #pragma unroll
                for (uint32_t nt = 0; nt < GQA_BIAS_W / 8u; nt++) {
                    const int d = dbase + (int)(8u * nt + gr);
                    const bool in = d >= 0 && d < (int)a.extent;
                    float acc[4] = {0.f, 0.f, 0.f, 0.f};
                    const uint16_t *pc = pr + (in ? d : 0);
                    const uint32_t b0 = in ? (uint32_t)pc[(2u * c) * a.extent] | ((uint32_t)pc[(2u * c + 1u) * a.extent] << 16) : 0u;
                    const uint32_t b1 = in ? (uint32_t)pc[(2u * c + 8u) * a.extent] | ((uint32_t)pc[(2u * c + 9u) * a.extent] << 16) : 0u;
                    gqa_mma(acc, ra[0], ra[1], ra[2], ra[3], b0, b1, bad);
                    #pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const int hh = e >> 1;
                        const float v = gqa_round(__fmul_rn(gqa_round(acc[e]), mtau[hh]));
                        bs[(w * 16u + gr + 8u * hh) * GQA_BIAS_W + 8u * nt + 2u * c + (e & 1)] = __float2bfloat16_rn(v);
                    }
                }
                gqa_nf_raise(a, bad);
                __syncwarp();
            }

            /* ---- S = Q K^T for the warp's 16 rows x 64 keys */
            float sc[8][4];
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
            #pragma unroll
            for (int kk = 0; kk < KS; kk++) {
                #pragma unroll
                for (int nt = 0; nt < 8; nt += 2) {
                    uint32_t kb[4];
                    gqa_ldsm4(kb, &ks[(8u * nt + (lane & 7u) + ((lane >> 4) << 3)) * P + kk * 16 + ((lane >> 3) & 1u) * 8u]);
                    gqa_mma_op(sc[nt], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], kb[0], kb[1]);
                    gqa_mma_op(sc[nt + 1], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], kb[2], kb[3]);
                }
            }
            /* scores in the base-2 domain: ((q . k) * scale + bias) * log2(e); masked keys -inf */
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int hh = e >> 1;
                    const uint32_t kx = 8u * nt + 2u * c + (e & 1), x = x0 + kx;
                    const bool valid = mval[hh] && x >= mlo[hh] && x <= mhi[hh];
                    float t = __fmul_rn(sc[nt][e], a.scale);
                    if (BIAS && bias_live && valid) {
                        t = __fadd_rn(t, __bfloat162float(bs[(w * 16u + gr + 8u * hh) * GQA_BIAS_W + (mpos[hh] - wpos0) + (GQA_SPLIT - 1u) - kx]));
                    }
                    sc[nt][e] = valid ? __fmul_rn(t, GQA_LOG2E) : -INFINITY;
                }
            }
            /* the online softmax: per row, the split's max, the rescale, P in a fixed order */
            #pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                float mx = -INFINITY;
                #pragma unroll
                for (int nt = 0; nt < 8; nt++) mx = fmaxf(mx, fmaxf(sc[nt][2 * hh], sc[nt][2 * hh + 1]));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
                const float mn = fmaxf(M[hh], mx);
                const bool live = mn != -INFINITY;
                const float c1 = live ? exp2f(__fsub_rn(M[hh], mn)) : 1.0f;
                float ls = 0.f;
                #pragma unroll
                for (int nt = 0; nt < 8; nt++) {
                    #pragma unroll
                    for (int e = 2 * hh; e < 2 * hh + 2; e++) {
                        const float p = live ? exp2f(__fsub_rn(sc[nt][e], mn)) : 0.f;
                        sc[nt][e] = p;
                        ls = __fadd_rn(ls, p);
                    }
                }
                ls = __fadd_rn(ls, __shfl_xor_sync(0xffffffffu, ls, 1));
                ls = __fadd_rn(ls, __shfl_xor_sync(0xffffffffu, ls, 2));
                L[hh] = __fadd_rn(__fmul_rn(L[hh], c1), ls);
                #pragma unroll
                for (int nd = 0; nd < ND; nd++) {
                    A[nd][2 * hh] = __fmul_rn(A[nd][2 * hh], c1);
                    A[nd][2 * hh + 1] = __fmul_rn(A[nd][2 * hh + 1], c1);
                }
                M[hh] = mn;
            }
            /* ---- O += P V, P as a bf16 hi + lo pair (two MMAs) */
            #pragma unroll
            for (int kq = 0; kq < 4; kq++) {
                uint32_t ah[4], al[4];
                gqa_split2(sc[2 * kq][0], sc[2 * kq][1], ah[0], al[0]);
                gqa_split2(sc[2 * kq][2], sc[2 * kq][3], ah[1], al[1]);
                gqa_split2(sc[2 * kq + 1][0], sc[2 * kq + 1][1], ah[2], al[2]);
                gqa_split2(sc[2 * kq + 1][2], sc[2 * kq + 1][3], ah[3], al[3]);
                #pragma unroll
                for (int j = 0; j < 4; j++) { ah[j] = gqa_op(ah[j]); al[j] = gqa_op(al[j]); }   /* once for every np */
                #pragma unroll
                for (int np = 0; np < ND / 2; np++) {
                    uint32_t vb[4];
                    gqa_ldsm4t(vb, &vs[(16u * kq + (lane & 7u) + ((lane >> 3) & 1u) * 8u) * P + 16 * np + (lane >> 4) * 8u]);
                    gqa_mma_op(A[2 * np], ah[0], ah[1], ah[2], ah[3], vb[0], vb[1]);
                    gqa_mma_op(A[2 * np], al[0], al[1], al[2], al[3], vb[0], vb[1]);
                    gqa_mma_op(A[2 * np + 1], ah[0], ah[1], ah[2], ah[3], vb[2], vb[3]);
                    gqa_mma_op(A[2 * np + 1], al[0], al[1], al[2], al[3], vb[2], vb[3]);
                }
            }
        }

        /* ---- the chunk is reduced: fold it (FOLD) or hand it to the combine */
        if (!warp_live) continue;
        #pragma unroll
        for (int hh = 0; hh < 2; hh++) {
            if (!mval[hh] || M[hh] == -INFINITY) continue;   /* the row has no key in this chunk */
            if constexpr (FOLD) {
                #pragma unroll
                for (int nd = 0; nd < ND; nd++) {
                    float m1 = RM[hh], l1 = RLs[hh];
                    gqa_fold(m1, l1, RA[nd][2 * hh], M[hh], L[hh], A[nd][2 * hh]);
                    m1 = RM[hh]; l1 = RLs[hh];
                    gqa_fold(m1, l1, RA[nd][2 * hh + 1], M[hh], L[hh], A[nd][2 * hh + 1]);
                }
                float a0 = 0.f;
                gqa_fold(RM[hh], RLs[hh], a0, M[hh], L[hh], 0.f);
            } else {
                const uint64_t slot = ((uint64_t)mrow[hh] * a.n_q + mhead[hh]) * a.max_chunks + (ch - mch0[hh]);
                float *dst = a.part + slot * HD;
                #pragma unroll
                for (int nd = 0; nd < ND; nd++) {
                    *reinterpret_cast<float2 *>(&dst[8 * nd + 2 * c]) = make_float2(A[nd][2 * hh], A[nd][2 * hh + 1]);
                }
                if (c == 0u) a.ml[slot] = make_float2(M[hh], L[hh]);
            }
        }
    }

    if constexpr (FOLD) {
        if (!warp_live) return;
        #pragma unroll
        for (int hh = 0; hh < 2; hh++) {
            if (!mval[hh]) continue;
            const uint64_t col0 = (uint64_t)mhead[hh] * HD;
            const __nv_bfloat16 *gate = a.gate ? a.gate + mrow[hh] * a.ld_gate + (uint64_t)mhead[hh] * a.gate_hs : nullptr;
            #pragma unroll
            for (int nd = 0; nd < ND; nd++) {
                const uint32_t d = 8u * nd + 2u * c;
                const float y0 = gqa_epilogue(RA[nd][2 * hh], RLs[hh], gate ? gate + d : nullptr);
                const float y1 = gqa_epilogue(RA[nd][2 * hh + 1], RLs[hh], gate ? gate + d + 1u : nullptr);
                *reinterpret_cast<__nv_bfloat162 *>(&a.out[mrow[hh] * a.ld_out + col0 + d]) = __floats2bfloat162_rn(y0, y1);
                if (a.tap) {
                    *reinterpret_cast<float2 *>(&a.tap[(uint64_t)mrow[hh] * a.n_q * HD + col0 + d]) = make_float2(y0, y1);
                }
            }
        }
    }
}

/* ---- the wide heads (L338): head_dim 256 / 512 -------------------------------------------------------------------
 * The 64 / 128 tile keeps a head's whole Q in registers (KS x 4), its whole output in registers (ND x 4, twice when
 * folding) and a 64-key K and V tile of the whole head in static shared memory: at 256 that is 67.6 KB of tiles and ~320
 * registers a thread, at 512 twice that -- past Turing's 64 KB and 255.  The wide kernel keeps the SAME arithmetic per
 * output element with two changes of staging:
 *   S = Q K^T in GQA_WIDE_D-dim PASSES: pass p stages K's dims [p D, p D + D) of the split and loads Q's fragments of
 *     those dims, and accumulates into the same score fragments -- the MMA k-steps run in the order kk = 0 .. HD/16 - 1
 *     exactly as one pass over the whole head would;
 *   O = P V over ONE D-wide slice of the output dims per CTA (blockIdx.y = g * HD / D + slice): an output element's
 *     P V accumulation reads only its own column, so it does not depend on which slice computes it; every slice CTA
 *     recomputes the same S, the same online softmax and the same (m, l) in the same order (QK^T is HD / D times
 *     redundant -- the price of the registers).  Slice 0 alone writes a chunk's (m, l) (the others hold the same
 *     values).
 * Gemma 4 (L338) declares nothing else: no relative bias, no output gate, no block run at these widths (refused). */
/// Output dims (and QK^T dims per pass) of one wide-head CTA.
constexpr uint32_t GQA_WIDE_D = 128u;

/// The attention over one tile x KV head g x output slice: as gqa_attn_kernel (no bias, no gate) with S accumulated in
/// GQA_WIDE_D-dim passes and O kept for one GQA_WIDE_D-wide slice.  Grid (FOLD ? 1 : grid_chunks, n_kv * HD /
/// GQA_WIDE_D, n_tiles), 128 threads, 34.8 KB static shared memory.
/// @tparam HD head_dim (256 or 512).
/// @tparam FOLD the folded schedule.
/// @tparam E4 the KV rows are E4M3 (widened to bf16 in the tile loader).
/// @param a the arguments.
template <int HD, bool FOLD, bool E4>
__global__ void __launch_bounds__(128) gqa_attn_wide_kernel(const gqa_args a) {
    constexpr int D = (int)GQA_WIDE_D;
    constexpr int NSL = HD / D;    /* output slices = QK^T passes */
    constexpr int P = D + 8;       /* bf16 smem pitch: 16-byte rows, conflict-free ldmatrix */
    constexpr int ND = D / 8;      /* PV n-tiles of the slice */
    constexpr int KS = D / 16;     /* QK k-steps per pass */
    constexpr int V8 = D / 8;      /* uint4 per staged head row */
    static_assert(HD % D == 0 && NSL >= 2, "a wide head is a whole number of slices");
    __shared__ __align__(16) __nv_bfloat16 ks[GQA_SPLIT * P];
    __shared__ __align__(16) __nv_bfloat16 vs[GQA_SPLIT * P];

    const gqa_tile T = a.tiles[blockIdx.z];
    const uint32_t g = blockIdx.y / NSL, sl = blockIdx.y % NSL;
    const gqa_row R0 = a.rows[T.row0];
    const gqa_row RL = a.rows[T.row0 + T.n - 1u];
    const uint32_t c_last = RL.hi / GQA_CHUNK;
    const uint32_t c_first = FOLD ? R0.chunk0 : R0.chunk0 + blockIdx.x;
    if (c_first > c_last) return;
    const uint32_t c_end = FOLD ? c_last : c_first;
    const uint32_t tid = threadIdx.x, w = tid >> 5, lane = tid & 31u, gr = lane >> 2, c = lane & 3u;
    const uint32_t G = a.G, mrows = T.n * G;

    /* this lane's two MMA rows: gr and gr + 8 of the warp's 16 */
    bool mval[2];
    uint32_t mhi[2], mlo[2], mrow[2], mhead[2], mch0[2];
    #pragma unroll
    for (int hh = 0; hh < 2; hh++) {
        const uint32_t m = 16u * w + gr + 8u * hh;
        mval[hh] = m < mrows;
        const uint32_t ri = mval[hh] ? m / G : 0u;
        const gqa_row &Rm = a.rows[T.row0 + ri];
        mhi[hh] = Rm.hi; mlo[hh] = Rm.lo; mch0[hh] = Rm.chunk0;
        mrow[hh] = T.row0 + ri;
        mhead[hh] = g * G + (mval[hh] ? m % G : 0u);
    }
    const bool warp_live = 16u * w < mrows;

    constexpr uint32_t ESZ = E4 ? 1u : 2u;   /* a KV element's bytes (E4M3 rows widen to bf16 in the loader) */
    const uint32_t koff = g * HD * ESZ, voff = ((a.n_kv + g) * HD + sl * D) * ESZ;
    float RM[2] = {-INFINITY, -INFINITY}, RLs[2] = {0.f, 0.f};
    float RA[FOLD ? ND : 1][4];
    #pragma unroll
    for (int nd = 0; nd < (FOLD ? ND : 1); nd++) RA[nd][0] = RA[nd][1] = RA[nd][2] = RA[nd][3] = 0.f;

    /* one D-wide K or V tile of the split's 64 keys: dims [d0, d0 + D) of the head at byte offset off */
    auto stage = [&](__nv_bfloat16 *dst, uint32_t off, uint32_t x0) {
        constexpr uint32_t LPT = GQA_SPLIT * V8 / 128u;
        uint4 ld[LPT];
        #pragma unroll
        for (uint32_t it = 0; it < LPT; it++) {
            const uint32_t i = tid + 128u * it, t = i / V8, part = i % V8, x = x0 + t;
            ld[it] = make_uint4(0u, 0u, 0u, 0u);
            if (x >= R0.lo && x <= RL.hi) {
                const uint8_t *rec = x >= R0.p0 ? a.fresh + (uint64_t)(R0.fresh0 + (x - R0.p0)) * a.token_bytes
                                                : R0.cache + (uint64_t)(R0.ring ? x % R0.cap : x) * a.token_bytes;
                if constexpr (E4) ld[it] = gqa_e4m3x8_to_bf16(*reinterpret_cast<const uint2 *>(rec + off + part * 8u));
                else              ld[it] = *reinterpret_cast<const uint4 *>(rec + off + part * 16u);
            }
        }
        uint32_t bad = 0u;   /* O59: a widened K / V element outside fp16's range (sm_75) */
        #pragma unroll
        for (uint32_t it = 0; it < LPT; it++) {
            const uint32_t i = tid + 128u * it, t = i / V8, part = i % V8;
            *reinterpret_cast<uint4 *>(&dst[t * P + part * 8u]) = gqa_opc4(ld[it], bad);   /* operand form */
        }
        gqa_nf_raise(a, bad);
    };

    for (uint32_t ch = c_first; ch <= c_end; ch++) {
        float M[2] = {-INFINITY, -INFINITY}, L[2] = {0.f, 0.f};
        float A[ND][4];
        #pragma unroll
        for (int nd = 0; nd < ND; nd++) A[nd][0] = A[nd][1] = A[nd][2] = A[nd][3] = 0.f;

        for (uint32_t s = 0; s < GQA_CHUNK_SPLITS; s++) {
            const uint32_t x0 = ch * GQA_CHUNK + s * GQA_SPLIT;
            if (x0 > RL.hi) break;
            if (x0 + GQA_SPLIT <= R0.lo) continue;

            /* ---- S = Q K^T for the warp's 16 rows x 64 keys, in NSL passes of D dims (k order unchanged) */
            float sc[8][4];
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
            for (int p = 0; p < NSL; p++) {
                __syncthreads();                   /* the previous pass / split is done with ks */
                stage(ks, koff + (uint32_t)p * D * ESZ, x0);
                __syncthreads();
                if (!warp_live) continue;
                uint32_t qa[KS][4];
                uint32_t qbad = 0u;   /* O59: a widened q element outside fp16's range (sm_75) */
                #pragma unroll
                for (int hh = 0; hh < 2; hh++) {
                    const __nv_bfloat16 *qp = a.q + ((uint64_t)mrow[hh] * a.n_q + mhead[hh]) * HD + p * D;
                    #pragma unroll
                    for (int kk = 0; kk < KS; kk++) {
                        qa[kk][hh]     = gqa_opc(mval[hh] ? *reinterpret_cast<const uint32_t *>(qp + kk * 16 + 2u * c) : 0u, qbad);
                        qa[kk][2 + hh] = gqa_opc(mval[hh] ? *reinterpret_cast<const uint32_t *>(qp + kk * 16 + 8u + 2u * c) : 0u, qbad);
                    }
                }
                gqa_nf_raise(a, qbad);
                #pragma unroll
                for (int kk = 0; kk < KS; kk++) {
                    #pragma unroll
                    for (int nt = 0; nt < 8; nt += 2) {
                        uint32_t kb[4];
                        gqa_ldsm4(kb, &ks[(8u * nt + (lane & 7u) + ((lane >> 4) << 3)) * P + kk * 16 + ((lane >> 3) & 1u) * 8u]);
                        gqa_mma_op(sc[nt], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], kb[0], kb[1]);
                        gqa_mma_op(sc[nt + 1], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], kb[2], kb[3]);
                    }
                }
            }
            /* the V tile of this CTA's output slice (ks is free after the last pass; vs after the previous split) */
            stage(vs, voff, x0);
            __syncthreads();
            if (!warp_live) continue;

            /* scores in the base-2 domain: (q . k) * scale * log2(e); masked keys -inf */
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int hh = e >> 1;
                    const uint32_t x = x0 + 8u * nt + 2u * c + (e & 1);
                    const bool valid = mval[hh] && x >= mlo[hh] && x <= mhi[hh];
                    sc[nt][e] = valid ? __fmul_rn(__fmul_rn(sc[nt][e], a.scale), GQA_LOG2E) : -INFINITY;
                }
            }
            /* the online softmax: per row, the split's max, the rescale, P in a fixed order */
            #pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                float mx = -INFINITY;
                #pragma unroll
                for (int nt = 0; nt < 8; nt++) mx = fmaxf(mx, fmaxf(sc[nt][2 * hh], sc[nt][2 * hh + 1]));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
                const float mn = fmaxf(M[hh], mx);
                const bool live = mn != -INFINITY;
                const float c1 = live ? exp2f(__fsub_rn(M[hh], mn)) : 1.0f;
                float ls = 0.f;
                #pragma unroll
                for (int nt = 0; nt < 8; nt++) {
                    #pragma unroll
                    for (int e = 2 * hh; e < 2 * hh + 2; e++) {
                        const float pv = live ? exp2f(__fsub_rn(sc[nt][e], mn)) : 0.f;
                        sc[nt][e] = pv;
                        ls = __fadd_rn(ls, pv);
                    }
                }
                ls = __fadd_rn(ls, __shfl_xor_sync(0xffffffffu, ls, 1));
                ls = __fadd_rn(ls, __shfl_xor_sync(0xffffffffu, ls, 2));
                L[hh] = __fadd_rn(__fmul_rn(L[hh], c1), ls);
                #pragma unroll
                for (int nd = 0; nd < ND; nd++) {
                    A[nd][2 * hh] = __fmul_rn(A[nd][2 * hh], c1);
                    A[nd][2 * hh + 1] = __fmul_rn(A[nd][2 * hh + 1], c1);
                }
                M[hh] = mn;
            }
            /* ---- O[slice] += P V[slice], P as a bf16 hi + lo pair (two MMAs) */
            #pragma unroll
            for (int kq = 0; kq < 4; kq++) {
                uint32_t ah[4], al[4];
                gqa_split2(sc[2 * kq][0], sc[2 * kq][1], ah[0], al[0]);
                gqa_split2(sc[2 * kq][2], sc[2 * kq][3], ah[1], al[1]);
                gqa_split2(sc[2 * kq + 1][0], sc[2 * kq + 1][1], ah[2], al[2]);
                gqa_split2(sc[2 * kq + 1][2], sc[2 * kq + 1][3], ah[3], al[3]);
                #pragma unroll
                for (int j = 0; j < 4; j++) { ah[j] = gqa_op(ah[j]); al[j] = gqa_op(al[j]); }   /* once for every np */
                #pragma unroll
                for (int np = 0; np < ND / 2; np++) {
                    uint32_t vb[4];
                    gqa_ldsm4t(vb, &vs[(16u * kq + (lane & 7u) + ((lane >> 3) & 1u) * 8u) * P + 16 * np + (lane >> 4) * 8u]);
                    gqa_mma_op(A[2 * np], ah[0], ah[1], ah[2], ah[3], vb[0], vb[1]);
                    gqa_mma_op(A[2 * np], al[0], al[1], al[2], al[3], vb[0], vb[1]);
                    gqa_mma_op(A[2 * np + 1], ah[0], ah[1], ah[2], ah[3], vb[2], vb[3]);
                    gqa_mma_op(A[2 * np + 1], al[0], al[1], al[2], al[3], vb[2], vb[3]);
                }
            }
        }

        /* ---- the chunk is reduced: fold it (FOLD) or hand the slice to the combine */
        if (!warp_live) continue;
        #pragma unroll
        for (int hh = 0; hh < 2; hh++) {
            if (!mval[hh] || M[hh] == -INFINITY) continue;   /* the row has no key in this chunk */
            if constexpr (FOLD) {
                #pragma unroll
                for (int nd = 0; nd < ND; nd++) {
                    float m1 = RM[hh], l1 = RLs[hh];
                    gqa_fold(m1, l1, RA[nd][2 * hh], M[hh], L[hh], A[nd][2 * hh]);
                    m1 = RM[hh]; l1 = RLs[hh];
                    gqa_fold(m1, l1, RA[nd][2 * hh + 1], M[hh], L[hh], A[nd][2 * hh + 1]);
                }
                float a0 = 0.f;
                gqa_fold(RM[hh], RLs[hh], a0, M[hh], L[hh], 0.f);
            } else {
                const uint64_t slot = ((uint64_t)mrow[hh] * a.n_q + mhead[hh]) * a.max_chunks + (ch - mch0[hh]);
                float *dst = a.part + slot * HD + sl * D;
                #pragma unroll
                for (int nd = 0; nd < ND; nd++) {
                    *reinterpret_cast<float2 *>(&dst[8 * nd + 2 * c]) = make_float2(A[nd][2 * hh], A[nd][2 * hh + 1]);
                }
                if (c == 0u && sl == 0u) a.ml[slot] = make_float2(M[hh], L[hh]);
            }
        }
    }

    if constexpr (FOLD) {
        if (!warp_live) return;
        #pragma unroll
        for (int hh = 0; hh < 2; hh++) {
            if (!mval[hh]) continue;
            const uint64_t col0 = (uint64_t)mhead[hh] * HD + sl * D;
            #pragma unroll
            for (int nd = 0; nd < ND; nd++) {
                const uint32_t d = 8u * nd + 2u * c;
                const float y0 = gqa_epilogue(RA[nd][2 * hh], RLs[hh], nullptr);
                const float y1 = gqa_epilogue(RA[nd][2 * hh + 1], RLs[hh], nullptr);
                *reinterpret_cast<__nv_bfloat162 *>(&a.out[mrow[hh] * a.ld_out + col0 + d]) = __floats2bfloat162_rn(y0, y1);
                if (a.tap) {
                    *reinterpret_cast<float2 *>(&a.tap[(uint64_t)mrow[hh] * a.n_q * HD + col0 + d]) = make_float2(y0, y1);
                }
            }
        }
    }
}

/* ---- L338: the decode-shaped call's attention: flash-decoding on the CUDA cores ---------------------------------- */
/// Threads of the decode attention CTA: one key of the chunk each (scores), 8 warps over its keys (P V).
constexpr uint32_t GQA_DEC_THREADS = 256u;
static_assert(GQA_DEC_THREADS == GQA_CHUNK, "the decode CTA scores one chunk, a key a thread");
/// Rows a decode CTA may attend together (pulsar_gqa_io::decode_group).
constexpr uint32_t GQA_DEC_GROUP_MAX = 4u;
/// Query vectors (rows x heads) a decode CTA attends at most (the static smem: q, P and the warp slab).
constexpr uint32_t GQA_DEC_QV_MAX = 8u;
/// The attention of RB consecutive rows x HB query heads of one KV head x one GQA_CHUNK-key chunk into the split
/// schedule's partials (the combine folds the chunks).  For each (row, head) -- a query vector: the 256 scores a key
/// a thread (q . k in f32 over 16-byte K loads, q in smem), the chunk's max and sum by fixed-order block reductions,
/// then O = P V with warp w taking keys w, w + 8, ... and each lane HD / 32 dims (16-byte V loads), the warps summed in
/// warp order -- the same operations in the same order whatever RB and HB are, so a (row, head)'s bytes never depend
/// on what rides with it.  The K vector a thread scores is loaded once for every head (and, when the rows read one
/// sequence's keys from the same first chunk -- a verify's -- for every row); P V runs the query vectors in passes of
/// PB (the accumulators' registers), each V vector loaded once a pass for its vectors.  Each row's chunk is its own
/// chunk0 + blockIdx.y, as one row a CTA.  Grid (n_q / HB, grid_chunks, ceil(n_rows / RB)), 256 threads: the heads
/// fastest, so whatever a KV head's heads are split into reads a chunk together.  (L338: one head a CTA re-read every
/// key G times -- 1 GB of L2 traffic for a 32K-key global layer, 1.7 ms on the RTX 5000; the wide-head MMA kernel
/// spends 62 of its 64 MMA rows on padding for a decode row.)
/// @tparam HD head_dim (a multiple of 256: one 16-byte load a lane per 8 dims).
/// @tparam E4 the KV rows are E4M3 (widened to bf16 as loaded).
/// @tparam RB rows a CTA attends (1 .. GQA_DEC_GROUP_MAX).
/// @tparam HB query heads a CTA attends (a divisor of the layer's G; RB x HB <= GQA_DEC_QV_MAX).
/// @param a the arguments (part / ml written; out untouched -- the combine writes it).
/// @param n_rows the call's rows.
template <int HD, bool E4, int RB, int HB>
__global__ void __launch_bounds__(GQA_DEC_THREADS) gqa_decode_kernel(const gqa_args a, const uint32_t n_rows) {
    constexpr uint32_t ESZ = E4 ? 1u : 2u;
    constexpr int DPL = HD / 32;                  /* dims a lane owns in P V */
    constexpr int V8 = HD / 8;                    /* 8-dim groups of a head row */
    constexpr int QV = RB * HB;                   /* query vectors: qv = r * HB + hh */
    constexpr int PB = HD >= 512 ? (QV < 2 ? QV : 2) : (QV < 4 ? QV : 4);   /* P V's vectors a pass */
    static_assert(HD % 256 == 0, "the decode attention's lane layout needs HD a multiple of 256");
    static_assert(RB >= 1 && RB <= (int)GQA_DEC_GROUP_MAX && HB >= 1 && QV <= (int)GQA_DEC_QV_MAX && QV % PB == 0,
                  "query vectors a decode CTA attends");
    __shared__ __align__(16) __nv_bfloat16 sq[QV][HD];
    __shared__ float sp[QV][GQA_CHUNK];
    __shared__ float sred[QV][GQA_DEC_THREADS / 32];
    __shared__ float sacc[GQA_DEC_THREADS / 32][HD];
    const uint32_t r0 = blockIdx.z * (uint32_t)RB, h0 = blockIdx.x * (uint32_t)HB, kc = blockIdx.y;
    const uint32_t tid = threadIdx.x, w = tid >> 5, lane = tid & 31u;
    const uint32_t nr = n_rows - r0 < (uint32_t)RB ? n_rows - r0 : (uint32_t)RB;
    gqa_row R[RB];
#pragma unroll
    for (int r = 0; r < RB; r++) R[r] = a.rows[r0 + ((uint32_t)r < nr ? (uint32_t)r : 0u)];
    uint32_t x0[RB];                              /* each row's chunk's first key */
    bool live[RB];                                /* the row has keys in this chunk */
    bool any = false, shared_kv = true;
#pragma unroll
    for (int r = 0; r < RB; r++) {
        const uint32_t ch = R[r].chunk0 + kc;
        x0[r] = ch * GQA_CHUNK;
        live[r] = (uint32_t)r < nr && ch <= R[r].hi / GQA_CHUNK;
        any = any || live[r];
        if ((uint32_t)r < nr)
            shared_kv = shared_kv && R[r].cache == R[0].cache && R[r].ring == R[0].ring && R[r].cap == R[0].cap &&
                        R[r].p0 == R[0].p0 && R[r].fresh0 == R[0].fresh0 && R[r].chunk0 == R[0].chunk0;
    }
    if (!any) return;                             /* past every row's last chunk (a bounded grid) */
    const uint32_t g = h0 / a.G;                  /* HB divides G: one KV head */
    const uint32_t koff = g * HD * ESZ, voff = (a.n_kv + g) * HD * ESZ;
#pragma unroll
    for (int r = 0; r < RB; r++)
        if (live[r])
#pragma unroll
            for (int hh = 0; hh < HB; hh++)
                for (uint32_t i = tid; i < (uint32_t)HD / 8u; i += GQA_DEC_THREADS)
                    reinterpret_cast<uint4 *>(sq[r * HB + hh])[i] =
                        reinterpret_cast<const uint4 *>(a.q + ((uint64_t)(r0 + r) * a.n_q + h0 + hh) * HD)[i];
    __syncthreads();
    auto rec_of = [&](const gqa_row &Rr, uint32_t x) -> const uint8_t * {
        return x >= Rr.p0 ? a.fresh + (uint64_t)(Rr.fresh0 + (x - Rr.p0)) * a.token_bytes
                          : Rr.cache + (uint64_t)(Rr.ring ? x % Rr.cap : x) * a.token_bytes;
    };
    auto ld8 = [&](const uint8_t *p) -> uint4 {   /* 8 elements as 8 bf16 */
        if constexpr (E4) return gqa_e4m3x8_to_bf16(*reinterpret_cast<const uint2 *>(p));
        else return *reinterpret_cast<const uint4 *>(p);
    };
    /* 1. the scores, a key a thread, in the base-2 domain; masked keys -inf */
    bool in[RB];
    float d[QV], s[QV];
#pragma unroll
    for (int r = 0; r < RB; r++) {
        const uint32_t x = x0[r] + tid;
        in[r] = live[r] && x >= R[r].lo && x <= R[r].hi;
    }
#pragma unroll
    for (int q = 0; q < QV; q++) d[q] = 0.f;
    const uint8_t *k0 = nullptr;
    if (shared_kv) {
        bool need = false;
#pragma unroll
        for (int r = 0; r < RB; r++) need = need || in[r];
        if (need) k0 = rec_of(R[0], x0[0] + tid) + koff;
    }
#pragma unroll 2
    for (int j = 0; j < V8; j++) {
        uint4 kv0 = make_uint4(0u, 0u, 0u, 0u);
        if (k0) kv0 = ld8(k0 + (size_t)j * 8u * ESZ);
#pragma unroll
        for (int r = 0; r < RB; r++) {
            if (!in[r]) continue;
            const uint4 kv = k0 ? kv0 : ld8(rec_of(R[r], x0[r] + tid) + koff + (size_t)j * 8u * ESZ);
            const uint32_t kw[4] = {kv.x, kv.y, kv.z, kv.w};
#pragma unroll
            for (int hh = 0; hh < HB; hh++) {
                const int q = r * HB + hh;
                const uint4 qv = reinterpret_cast<const uint4 *>(sq[q])[j];
                const uint32_t qw[4] = {qv.x, qv.y, qv.z, qv.w};
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    d[q] = __fmaf_rn(__uint_as_float(qw[e] << 16), __uint_as_float(kw[e] << 16), d[q]);
                    d[q] = __fmaf_rn(__uint_as_float(qw[e] & 0xffff0000u), __uint_as_float(kw[e] & 0xffff0000u), d[q]);
                }
            }
        }
    }
#pragma unroll
    for (int q = 0; q < QV; q++) s[q] = in[q / HB] ? __fmul_rn(__fmul_rn(d[q], a.scale), GQA_LOG2E) : -INFINITY;
    /* 2. each vector's chunk max and sum, fixed-order reductions (warps, then the 8 warp values in order) */
#pragma unroll
    for (int q = 0; q < QV; q++) {
        float mx = s[q];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        if (lane == 0) sred[q][w] = mx;
    }
    __syncthreads();
    float M[QV];
    bool qlive[QV];                               /* the vector has a key in this chunk */
    bool keyed = false;
#pragma unroll
    for (int q = 0; q < QV; q++) {
        M[q] = -INFINITY;
#pragma unroll
        for (uint32_t i = 0; i < GQA_DEC_THREADS / 32; i++) M[q] = fmaxf(M[q], sred[q][i]);
        qlive[q] = live[q / HB] && M[q] != -INFINITY;   /* none: the combine never reads its slot */
        keyed = keyed || qlive[q];
    }
    if (!keyed) return;
    float ls[QV];
#pragma unroll
    for (int q = 0; q < QV; q++) {
        const float pv = !qlive[q] || s[q] == -INFINITY ? 0.f : exp2f(__fsub_rn(s[q], M[q]));
        sp[q][tid] = pv;
        ls[q] = pv;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ls[q] = __fadd_rn(ls[q], __shfl_xor_sync(0xffffffffu, ls[q], o));
    }
    __syncthreads();                              /* sred's max reads are done */
#pragma unroll
    for (int q = 0; q < QV; q++)
        if (lane == 0) sred[q][w] = ls[q];
    __syncthreads();
    /* 3. O = P V in passes of PB vectors: warp w the chunk's keys w, w + 8, ...; lane the dims [lane * DPL, + DPL) */
#pragma unroll 1
    for (int pb = 0; pb < QV; pb += PB) {
        float acc[PB][DPL];
#pragma unroll
        for (int k = 0; k < PB; k++)
#pragma unroll
            for (int e = 0; e < DPL; e++) acc[k][e] = 0.f;
        for (uint32_t t = w; t < GQA_CHUNK; t += GQA_DEC_THREADS / 32) {
            float p[PB];
            bool need = false;
#pragma unroll
            for (int k = 0; k < PB; k++) {
                p[k] = qlive[pb + k] ? sp[pb + k][t] : 0.f;
                need = need || p[k] != 0.f;
            }
            if (!need) continue;                  /* a masked key (or one whose weights underflowed: adds 0) */
#pragma unroll
            for (int jj = 0; jj < DPL / 8; jj++) {
                uint4 vv = make_uint4(0u, 0u, 0u, 0u);
                int vr = -1;                      /* the row whose V vector vv holds */
#pragma unroll
                for (int k = 0; k < PB; k++) {
                    if (p[k] == 0.f) continue;
                    const int r = (pb + k) / HB;
                    if (vr < 0 || (!shared_kv && r != vr)) {
                        vv = ld8(rec_of(R[r], x0[r] + t) + voff + (size_t)lane * DPL * ESZ + (size_t)jj * 8u * ESZ);
                        vr = r;
                    }
                    const uint32_t vw[4] = {vv.x, vv.y, vv.z, vv.w};
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        acc[k][8 * jj + 2 * e] = __fmaf_rn(p[k], __uint_as_float(vw[e] << 16), acc[k][8 * jj + 2 * e]);
                        acc[k][8 * jj + 2 * e + 1] =
                            __fmaf_rn(p[k], __uint_as_float(vw[e] & 0xffff0000u), acc[k][8 * jj + 2 * e + 1]);
                    }
                }
            }
        }
        /* each vector's warps summed in warp order through the one smem slab */
#pragma unroll
        for (int k = 0; k < PB; k++) {
            const int q = pb + k;
            if (!qlive[q]) continue;              /* uniform: qlive is the block's */
#pragma unroll
            for (int e = 0; e < DPL; e++) sacc[w][lane * DPL + e] = acc[k][e];
            __syncthreads();
            const int r = q / HB, hh = q % HB;
            const uint64_t slot = ((uint64_t)(r0 + r) * a.n_q + h0 + hh) * a.max_chunks + kc;
            for (uint32_t dd = tid; dd < (uint32_t)HD; dd += GQA_DEC_THREADS) {
                float z = 0.f;
#pragma unroll
                for (uint32_t i = 0; i < GQA_DEC_THREADS / 32; i++) z = __fadd_rn(z, sacc[i][dd]);
                a.part[slot * HD + dd] = z;
            }
            if (tid == 0) {
                float L = 0.f;
#pragma unroll
                for (uint32_t i = 0; i < GQA_DEC_THREADS / 32; i++) L = __fadd_rn(L, sred[q][i]);
                a.ml[slot] = make_float2(M[q], L);
            }
            __syncthreads();                      /* sacc's reads are done before the next vector writes it */
        }
    }
}

/* ---- L338: the decode rows of a wide GQA (G a multiple of 8: Gemma's global layers) on the tensor cores ----------- */
/// Threads of the MMA decode CTA (8 warps: 32 keys each to score, 1/8 of the dims each in P V).
constexpr uint32_t GQA_DMMA_THREADS = 256u;
/// Query vectors a MMA decode CTA attends: one row x 8 heads of one KV head (the MMA's n).
constexpr uint32_t GQA_DMMA_NV = 8u;
/// Keys a V tile holds (one m16n8k16 k step).
constexpr uint32_t GQA_DMMA_VT = 16u;
/// RB rows x 8 query heads of one KV head x one GQA_CHUNK-key chunk into the split schedule's partials, on the
/// tensor cores: S^T = K Q^T (the keys the MMA's m, each row's 8 heads one n tile -- no padding), the online softmax
/// per (row, head) in base 2 (the chunk's max and sum, fixed-order: the thread's keys, the xor-shuffle over its quad's
/// lanes, the warps in order), then O^T = V^T P^T with P split hi + lo bf16 (gqa_split2: P to ~2^-16) and V through
/// 16-key smem tiles (ldmatrix .trans), the next tile's loads in flight while this one's MMAs run.  K is read
/// straight from the pool: each thread 32 contiguous bytes of a key row a 64-dim group, its A fragments those bytes in
/// order (the head dim's order inside the dot product is the fragment's; Q's B fragments, read from the workspace, the
/// same order) -- one load for every row's every head.  The rows a pass attends read one sequence's keys from the
/// same first chunk (a verify's, in one pass); a group whose rows do not runs them in passes, one sequence's rows a
/// pass.  On sm_75 the bf16 operands widen to fp16 exactly (gqa_mma).  Not the flash-decoding arm's bytes (graded,
/// L338: "take the speed"); a (row, head)'s bytes are its own at any RB -- an MMA column of its own, the chunks and
/// the warps' keys fixed -- so a verify's rows and a plain decode's are the same arm's bytes.  Grid (n_q / 8,
/// grid_chunks, ceil(n_rows / RB)), 256 threads, dynamic smem gqa_dmma_smem(HD, RB).  BF16 KV rows.
/// @tparam HD head_dim (a multiple of 64, at most 512).
/// @tparam RB rows a CTA (1, 2, 4).
/// @param a the arguments (part / ml written; out untouched -- the combine writes it).
/// @param n_rows the call's rows.
template <int HD, int RB>
__global__ void __launch_bounds__(GQA_DMMA_THREADS) gqa_decode_mma_kernel(const gqa_args a, const uint32_t n_rows) {
    constexpr int NV = (int)GQA_DMMA_NV, KEYS = (int)GQA_CHUNK, NW = (int)(GQA_DMMA_THREADS / 32u);
    constexpr int KPW = KEYS / NW;                /* keys a warp scores: 32 */
    constexpr int MT = KPW / 16;                  /* its m tiles */
    constexpr int QG = HD / 64;                   /* 64-dim groups of the dot product */
    constexpr int VT = (int)GQA_DMMA_VT, VP = HD + 8;   /* V tile keys, its smem pitch (conflict-free ldmatrix) */
    constexpr int DW = HD / NW, DT = DW / 16;     /* P V: dims a warp owns, its m tiles */
    constexpr int VL = VT * HD / 8 / (int)GQA_DMMA_THREADS;   /* 16-byte V loads a thread a tile */
    static_assert(HD % 64 == 0 && HD <= 512 && KPW % 16 == 0 && DW % 16 == 0 && VL >= 1 &&
                      VL * (int)GQA_DMMA_THREADS == VT * HD / 8 && (RB == 1 || RB == 2 || RB == 4),
                  "the MMA decode's tiling");
    extern __shared__ __align__(16) uint8_t gqa_dmma_smem_buf[];
    __nv_bfloat16 (*sv)[VP] = reinterpret_cast<__nv_bfloat16 (*)[VP]>(gqa_dmma_smem_buf);
    float (*sp)[KEYS] = reinterpret_cast<float (*)[KEYS]>(gqa_dmma_smem_buf + (size_t)VT * VP * 2u);   /* [RB * NV] */
    __shared__ float sred[NW][RB * NV];
    __shared__ float sM[RB * NV], sL[RB * NV];
    const uint32_t r0 = blockIdx.z * (uint32_t)RB, h0 = blockIdx.x * (uint32_t)NV, kc = blockIdx.y;
    const uint32_t tid = threadIdx.x, w = tid >> 5, lane = tid & 31u, gq = lane >> 2, t = lane & 3u;
    const uint32_t nr = n_rows - r0 < (uint32_t)RB ? n_rows - r0 : (uint32_t)RB;
    gqa_row R[RB];
#pragma unroll
    for (int r = 0; r < RB; r++) R[r] = a.rows[r0 + ((uint32_t)r < nr ? (uint32_t)r : 0u)];
    const uint32_t g = h0 / a.G;
    const uint32_t koff = g * HD * 2u, voff = (a.n_kv + g) * HD * 2u;
    const uint32_t lk = (lane & 7u) + 8u * (lane >> 4), ld8 = 8u * ((lane >> 3) & 1u);   /* this lane's ldmatrix row */
    auto same_seq = [&](const gqa_row &x, const gqa_row &y) {
        return x.cache == y.cache && x.ring == y.ring && x.cap == y.cap && x.p0 == y.p0 && x.fresh0 == y.fresh0 &&
               x.chunk0 == y.chunk0;
    };
    uint32_t done = 0u;                           /* rows a pass has run */
#pragma unroll 1
    for (int rep = 0; rep < RB; rep++) {
        if ((uint32_t)rep >= nr || (done >> rep) & 1u) continue;
        /* this pass: the rows reading rep's sequence from its first chunk */
        bool act[RB];
        bool any = false;
#pragma unroll
        for (int r = 0; r < RB; r++) {
            act[r] = (uint32_t)r < nr && !((done >> r) & 1u) && same_seq(R[r], R[rep]);
            if (act[r]) done |= 1u << r;
            act[r] = act[r] && R[r].chunk0 + kc <= R[r].hi / GQA_CHUNK;   /* has keys in this chunk */
            any = any || act[r];
        }
        if (!any) continue;                       /* uniform: the rows table is the block's */
        const gqa_row &Q = R[rep];
        const uint32_t x0 = (Q.chunk0 + kc) * GQA_CHUNK;
        uint32_t lo = 0xffffffffu, hi = 0u;       /* the keys any active row attends */
#pragma unroll
        for (int r = 0; r < RB; r++)
            if (act[r]) {
                lo = R[r].lo < lo ? R[r].lo : lo;
                hi = R[r].hi > hi ? R[r].hi : hi;
            }
        auto rec_of = [&](uint32_t x) -> const uint8_t * {
            return x >= Q.p0 ? a.fresh + (uint64_t)(Q.fresh0 + (x - Q.p0)) * a.token_bytes
                             : Q.cache + (uint64_t)(Q.ring ? x % Q.cap : x) * a.token_bytes;
        };
        auto loaded = [&](uint32_t x) { return x >= lo && x <= hi; };
        /* 1. S^T = K Q^T: warp w the keys [KPW w, KPW w + KPW), m tile m its keys 16 m + (gq, gq + 8), n tile r row r */
        float S[MT][RB][4];
#pragma unroll
        for (int m = 0; m < MT; m++)
#pragma unroll
            for (int r = 0; r < RB; r++)
#pragma unroll
                for (int e = 0; e < 4; e++) S[m][r][e] = 0.f;
        const uint32_t kb = w * (uint32_t)KPW;
#pragma unroll 1
        for (int q = 0; q < QG; q++) {
            uint32_t qw[RB][8];
            uint32_t bad = 0u;   /* O59: a widened q / k element outside fp16's range (sm_75) */
#pragma unroll
            for (int r = 0; r < RB; r++) {
                const uint4 *qp = reinterpret_cast<const uint4 *>(
                    a.q + ((uint64_t)(r0 + (act[r] ? (uint32_t)r : 0u)) * a.n_q + h0 + gq) * HD + 64 * q + 16 * (int)t);
                const uint4 qa = gqa_opc4(qp[0], bad), qb = gqa_opc4(qp[1], bad);
                qw[r][0] = qa.x; qw[r][1] = qa.y; qw[r][2] = qa.z; qw[r][3] = qa.w;
                qw[r][4] = qb.x; qw[r][5] = qb.y; qw[r][6] = qb.z; qw[r][7] = qb.w;
            }
#pragma unroll
            for (int m = 0; m < MT; m++) {
                const uint32_t xa = x0 + kb + 16u * (uint32_t)m + gq, xb = xa + 8u;
                const size_t off = koff + (size_t)(64 * q + 16 * (int)t) * 2u;
                uint4 a0 = make_uint4(0u, 0u, 0u, 0u), a1 = a0, b0 = a0, b1 = a0;
                if (loaded(xa)) {
                    const uint4 *p = reinterpret_cast<const uint4 *>(rec_of(xa) + off);
                    a0 = p[0];
                    a1 = p[1];
                }
                if (loaded(xb)) {
                    const uint4 *p = reinterpret_cast<const uint4 *>(rec_of(xb) + off);
                    b0 = p[0];
                    b1 = p[1];
                }
                /* widened once for every row's MMAs (gqa_opc) */
                const uint32_t k0[8] = {gqa_opc(a0.x, bad), gqa_opc(a0.y, bad), gqa_opc(a0.z, bad), gqa_opc(a0.w, bad),
                                        gqa_opc(a1.x, bad), gqa_opc(a1.y, bad), gqa_opc(a1.z, bad), gqa_opc(a1.w, bad)};
                const uint32_t k1[8] = {gqa_opc(b0.x, bad), gqa_opc(b0.y, bad), gqa_opc(b0.z, bad), gqa_opc(b0.w, bad),
                                        gqa_opc(b1.x, bad), gqa_opc(b1.y, bad), gqa_opc(b1.z, bad), gqa_opc(b1.w, bad)};
#pragma unroll
                for (int r = 0; r < RB; r++) {
                    if (!act[r]) continue;
#pragma unroll
                    for (int jj = 0; jj < 4; jj++)
                        gqa_mma_op(S[m][r], k0[2 * jj], k1[2 * jj], k0[2 * jj + 1], k1[2 * jj + 1], qw[r][2 * jj],
                                qw[r][2 * jj + 1]);
                }
            }
            gqa_nf_raise(a, bad);
        }
        /* 2. each (row, head)'s chunk max and sum: the thread's heads 2t, 2t + 1 over its 2 MT keys, the quad lanes,
         *    the warps in order */
        float mx[RB][2];
#pragma unroll
        for (int r = 0; r < RB; r++) {
            mx[r][0] = mx[r][1] = -INFINITY;
#pragma unroll
            for (int m = 0; m < MT; m++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const uint32_t x = x0 + kb + 16u * (uint32_t)m + gq + (e >= 2 ? 8u : 0u);
                    const bool in = act[r] && x >= R[r].lo && x <= R[r].hi;
                    S[m][r][e] = in ? __fmul_rn(__fmul_rn(S[m][r][e], a.scale), GQA_LOG2E) : -INFINITY;
                    mx[r][e & 1] = fmaxf(mx[r][e & 1], S[m][r][e]);
                }
#pragma unroll
            for (int o = 4; o < 32; o <<= 1) {
                mx[r][0] = fmaxf(mx[r][0], __shfl_xor_sync(0xffffffffu, mx[r][0], o));
                mx[r][1] = fmaxf(mx[r][1], __shfl_xor_sync(0xffffffffu, mx[r][1], o));
            }
            if (gq == 0u) {
                sred[w][r * NV + 2 * (int)t] = mx[r][0];
                sred[w][r * NV + 2 * (int)t + 1] = mx[r][1];
            }
        }
        __syncthreads();
        float Mx[RB][2], ls[RB][2];
#pragma unroll
        for (int r = 0; r < RB; r++) {
            Mx[r][0] = Mx[r][1] = -INFINITY;
#pragma unroll
            for (int i = 0; i < NW; i++) {
                Mx[r][0] = fmaxf(Mx[r][0], sred[i][r * NV + 2 * (int)t]);
                Mx[r][1] = fmaxf(Mx[r][1], sred[i][r * NV + 2 * (int)t + 1]);
            }
            ls[r][0] = ls[r][1] = 0.f;
#pragma unroll
            for (int m = 0; m < MT; m++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const float M = Mx[r][e & 1];
                    const float p = S[m][r][e] == -INFINITY ? 0.f : exp2f(__fsub_rn(S[m][r][e], M));
                    const uint32_t key = kb + 16u * (uint32_t)m + gq + (e >= 2 ? 8u : 0u);
                    sp[r * NV + 2 * (int)t + (e & 1)][key] = p;
                    ls[r][e & 1] = __fadd_rn(ls[r][e & 1], p);
                }
#pragma unroll
            for (int o = 4; o < 32; o <<= 1) {
                ls[r][0] = __fadd_rn(ls[r][0], __shfl_xor_sync(0xffffffffu, ls[r][0], o));
                ls[r][1] = __fadd_rn(ls[r][1], __shfl_xor_sync(0xffffffffu, ls[r][1], o));
            }
        }
        __syncthreads();                          /* sred's max reads are done */
        if (gq == 0u)
#pragma unroll
            for (int r = 0; r < RB; r++) {
                sred[w][r * NV + 2 * (int)t] = ls[r][0];
                sred[w][r * NV + 2 * (int)t + 1] = ls[r][1];
            }
        __syncthreads();
        if (w == 0u && gq == 0u)
#pragma unroll
            for (int r = 0; r < RB; r++) {
                float L0 = 0.f, L1 = 0.f;
#pragma unroll
                for (int i = 0; i < NW; i++) {
                    L0 = __fadd_rn(L0, sred[i][r * NV + 2 * (int)t]);
                    L1 = __fadd_rn(L1, sred[i][r * NV + 2 * (int)t + 1]);
                }
                sM[r * NV + 2 * (int)t] = Mx[r][0]; sM[r * NV + 2 * (int)t + 1] = Mx[r][1];
                sL[r * NV + 2 * (int)t] = L0;       sL[r * NV + 2 * (int)t + 1] = L1;
            }
        /* 3. O^T = V^T P^T: warp w the dims [DW w, DW w + DW), 16-key V tiles through smem -- the next tile's loads in
         *    flight (registers) while this one's MMAs run; each V fragment serves every active row */
        float acc[DT][RB][4];
#pragma unroll
        for (int d = 0; d < DT; d++)
#pragma unroll
            for (int r = 0; r < RB; r++)
#pragma unroll
                for (int e = 0; e < 4; e++) acc[d][r][e] = 0.f;
        uint4 vr[VL];
        auto v_load = [&](int kt) {
#pragma unroll
            for (int j = 0; j < VL; j++) {
                const uint32_t i = tid + (uint32_t)j * GQA_DMMA_THREADS;
                const uint32_t key = i / (uint32_t)(HD / 8), c = i % (uint32_t)(HD / 8);
                const uint32_t x = x0 + (uint32_t)(kt * VT) + key;
                vr[j] = loaded(x) ? reinterpret_cast<const uint4 *>(rec_of(x) + voff)[c] : make_uint4(0u, 0u, 0u, 0u);
            }
        };
        v_load(0);
#pragma unroll 1
        for (int kt = 0; kt < KEYS / VT; kt++) {
            uint32_t vbad = 0u;                   /* O59: a widened V element outside fp16's range (sm_75) */
            __syncthreads();                      /* sp written (first tile); the last tile's ldmatrix reads done */
#pragma unroll
            for (int j = 0; j < VL; j++) {
                const uint32_t i = tid + (uint32_t)j * GQA_DMMA_THREADS;
                /* the V tile goes to smem widened (gqa_opc): its fragments serve every row */
                reinterpret_cast<uint4 *>(&sv[i / (uint32_t)(HD / 8)][0])[i % (uint32_t)(HD / 8)] = gqa_opc4(vr[j], vbad);
            }
            gqa_nf_raise(a, vbad);
            __syncthreads();
            if (kt + 1 < KEYS / VT) v_load(kt + 1);
            const uint32_t kk = (uint32_t)(kt * VT);
            uint32_t hi0[RB], lo0[RB], hi1[RB], lo1[RB];
#pragma unroll
            for (int r = 0; r < RB; r++) {
                gqa_split2(sp[r * NV + gq][kk + 2u * t], sp[r * NV + gq][kk + 2u * t + 1u], hi0[r], lo0[r]);
                gqa_split2(sp[r * NV + gq][kk + 8u + 2u * t], sp[r * NV + gq][kk + 8u + 2u * t + 1u], hi1[r], lo1[r]);
                hi0[r] = gqa_op(hi0[r]); lo0[r] = gqa_op(lo0[r]); hi1[r] = gqa_op(hi1[r]); lo1[r] = gqa_op(lo1[r]);
            }
#pragma unroll
            for (int d = 0; d < DT; d++) {
                uint32_t rr[4];
                gqa_ldsm4t(rr, &sv[lk][w * (uint32_t)DW + 16u * (uint32_t)d + ld8]);
#pragma unroll
                for (int r = 0; r < RB; r++) {
                    if (!act[r]) continue;
                    gqa_mma_op(acc[d][r], rr[0], rr[1], rr[2], rr[3], hi0[r], hi1[r]);
                    gqa_mma_op(acc[d][r], rr[0], rr[1], rr[2], rr[3], lo0[r], lo1[r]);
                }
            }
        }
        /* the partials: acc[d][r] holds dims (DW w + 16 d + gq [+ 8]) x row r's heads (2t, 2t + 1); a (row, head) with
         * no key here (max -inf) is never read by the combine */
#pragma unroll
        for (int r = 0; r < RB; r++) {
            if (!act[r]) continue;
#pragma unroll
            for (int d = 0; d < DT; d++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const uint32_t n = 2u * t + (uint32_t)(e & 1);
                    const uint32_t dim = w * (uint32_t)DW + 16u * (uint32_t)d + gq + (e >= 2 ? 8u : 0u);
                    const uint64_t slot = ((uint64_t)(r0 + r) * a.n_q + h0 + n) * a.max_chunks + kc;
                    a.part[slot * HD + dim] = acc[d][r][e];
                }
        }
        __syncthreads();                          /* sM / sL written; sp and sred free for the next pass */
        if (tid < (uint32_t)(RB * NV)) {
            const int r = (int)tid / NV;
            if (act[r]) {
                const uint64_t slot = ((uint64_t)(r0 + r) * a.n_q + h0 + tid % (uint32_t)NV) * a.max_chunks + kc;
                a.ml[slot] = make_float2(sM[tid], sL[tid]);
            }
        }
    }
}

/// Dynamic smem of gqa_decode_mma_kernel<HD, RB>: the V tile and the RB x 8 vectors' P.
/// @param hd head_dim
/// @param rb rows a CTA
/// @return bytes
constexpr size_t gqa_dmma_smem(uint32_t hd, uint32_t rb) {
    return (size_t)GQA_DMMA_VT * (hd + 8u) * 2u + (size_t)rb * GQA_DMMA_NV * GQA_CHUNK * 4u;
}

/// Launches gqa_decode_mma_kernel<HD, RB>, raising its dynamic smem limit on its first launch (one flag an
/// instantiation: the three row counts' kernels share a pointer type, not a limit).
/// @tparam HD head_dim.
/// @tparam RB rows a CTA.
/// @param grid the grid.
/// @param aa the arguments.
/// @param n_rows the call's rows.
/// @param st the stream.
template <int HD, int RB>
void gqa_decode_mma_go(dim3 grid, const gqa_args &aa, uint32_t n_rows, cudaStream_t st) {
    constexpr size_t bytes = gqa_dmma_smem(HD, RB);
    static bool raised = false;
    if (!raised) {
        (void)cudaFuncSetAttribute(gqa_decode_mma_kernel<HD, RB>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   (int)bytes);
        raised = true;
    }
    gqa_decode_mma_kernel<HD, RB><<<grid, GQA_DMMA_THREADS, bytes, st>>>(aa, n_rows);
}

/// Launches gqa_decode_mma_kernel at rb rows a CTA (1, 2, 4).
/// @tparam HD head_dim.
/// @param rb rows a CTA.
/// @param grid the grid.
/// @param aa the arguments.
/// @param n_rows the call's rows.
/// @param st the stream.
template <int HD>
void gqa_decode_mma_dispatch(uint32_t rb, dim3 grid, const gqa_args &aa, uint32_t n_rows, cudaStream_t st) {
    if (rb == 4u)      gqa_decode_mma_go<HD, 4>(grid, aa, n_rows, st);
    else if (rb == 2u) gqa_decode_mma_go<HD, 2>(grid, aa, n_rows, st);
    else               gqa_decode_mma_go<HD, 1>(grid, aa, n_rows, st);
}

/// An instrument's forced decode arm (pulsar_gqa_decode_arm_force): 0 the pick's, 1 flash-decoding, 2 the MMA arm.
int g_dec_arm_force = 0;

/// A flash-decoding CTA's shape: rows x query heads of one KV head.
struct gqa_dec_shape {
    uint32_t rb;   ///< rows (1, 2, 4)
    uint32_t hb;   ///< query heads (1, 2, 4, 8; a divisor of G)
};
/// Launches gqa_decode_kernel at one of its instantiated shapes: the pick's (1 x 1, 1 x 2, 1 x 4, 2 x 2).
/// @tparam HD head_dim.
/// @tparam E4 E4M3 KV rows.
/// @param rb rows a CTA.
/// @param hb heads a CTA.
/// @param grid the grid.
/// @param aa the arguments.
/// @param n_rows the call's rows.
/// @param st the stream.
template <int HD, bool E4>
void gqa_decode_dispatch(uint32_t rb, uint32_t hb, dim3 grid, const gqa_args &aa, uint32_t n_rows, cudaStream_t st) {
    switch (rb * 16u + hb) {
    case 0x11: gqa_decode_kernel<HD, E4, 1, 1><<<grid, GQA_DEC_THREADS, 0, st>>>(aa, n_rows); break;
    case 0x12: gqa_decode_kernel<HD, E4, 1, 2><<<grid, GQA_DEC_THREADS, 0, st>>>(aa, n_rows); break;
    case 0x14: gqa_decode_kernel<HD, E4, 1, 4><<<grid, GQA_DEC_THREADS, 0, st>>>(aa, n_rows); break;
    default:   gqa_decode_kernel<HD, E4, 2, 2><<<grid, GQA_DEC_THREADS, 0, st>>>(aa, n_rows); break;   /* 0x22 */
    }
}

/// An instrument's forced flash-decoding shape (pulsar_gqa_decode_force); {0, 0} = the pick's.
gqa_dec_shape g_dec_force = {0u, 0u};

/// The device's SM count (cached: the pick's parallelism floor).
/// @return SMs of the current device
uint32_t gqa_sm_count() {
    static int sms = 0;
    if (sms == 0) {
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) !=
                                                      cudaSuccess || sms <= 0)
            sms = 1;
    }
    return (uint32_t)sms;
}

/// The flash-decoding CTA shape a call runs at: an instrument's forced one, else the pick -- measured on the RTX 5000
/// (gqa_attn_bench GQA_BENCH_GEMMA: every shape at 512 .. 32K keys, 1 .. 16 rows, L338).  A narrow GQA (G <= 2,
/// Gemma's window layers: 1024 keys at most) shares every head, and from 6 rows (when the caller's rows are one
/// sequence's) two rows a CTA (16 rows: 512 -> 284 us).  A wider one (G = 8, the global layers) shares as many heads
/// as still leave a CTA a SM -- 4 at depth (a row at 32K keys: 1733 -> 959 us), fewer on a short context's few chunks
/// (512 keys: one head, 85 us vs 4 heads' 125) -- one row a CTA (its row groups lost to head sharing at every depth).
/// A function of the layer, the bounded chunk count and n_rows: what a captured step's key already holds.
/// @param G query heads a KV head.
/// @param n_q the layer's query heads.
/// @param chunks the call's chunk CTAs a (row, head) (the bounded grid's).
/// @param n_rows the call's rows.
/// @param cap_rows the caller's most rows a CTA (pulsar_gqa_io::decode_group, 1 .. 4).
/// @return the shape.
gqa_dec_shape gqa_dec_pick(uint32_t G, uint32_t n_q, uint32_t chunks, uint32_t n_rows, uint32_t cap_rows) {
    if (g_dec_force.rb) return g_dec_force;
    if (G <= 2u) return {n_rows >= 6u && cap_rows >= 2u && G == 2u ? 2u : 1u, G};
    for (uint32_t hb = 4u; hb > 1u; hb /= 2u)
        if (G % hb == 0u && (uint64_t)(n_q / hb) * chunks * n_rows >= gqa_sm_count()) return {1u, hb};
    return {1u, 1u};
}

/* ---- the split schedule's fold: grid (row, query head), thread = dim ------------ */
/// Folds each (row, query head)'s chunk partials in chunk order and runs the epilogue.  Grid (n_rows, n_q), HD threads.
/// @tparam HD head_dim; thread d owns output dim d.
/// @param a the arguments (part, ml, gate, out, tap).
template <int HD>
__global__ void __launch_bounds__(HD) gqa_combine_kernel(const gqa_args a) {
    const uint32_t row = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const gqa_row R = a.rows[row];
    const uint32_t nch = R.hi / GQA_CHUNK - R.chunk0 + 1u;
    const uint64_t base = ((uint64_t)row * a.n_q + h) * a.max_chunks;
    float M = -INFINITY, L = 0.f, A = 0.f;
    for (uint32_t i = 0; i < nch; i++) {
        const float2 e = a.ml[base + i];
        gqa_fold(M, L, A, e.x, e.y, a.part[(base + i) * HD + d]);
    }
    const __nv_bfloat16 *gate = a.gate ? a.gate + row * a.ld_gate + (uint64_t)h * a.gate_hs + d : nullptr;
    const float y = gqa_epilogue(A, L, gate);
    a.out[row * a.ld_out + (uint64_t)h * HD + d] = __float2bfloat16_rn(y);
    if (a.tap) a.tap[((uint64_t)row * a.n_q + h) * HD + d] = y;
}

/* ---- the call's KV rows into the pools ------------------------------------------- */
/// Copies each fresh KV row to its pool slot (ring: pos % cap; a ring slot a later row of the call takes is skipped).
/// Grid n_rows, 128 threads, uint4 copies: token_bytes is a multiple of 16 and pools are 16-byte aligned.
/// @param rows the row descriptors.
/// @param fresh the call's KV rows, [n_rows][token_bytes].
/// @param token_bytes bytes per KV row.
__global__ void __launch_bounds__(128) gqa_commit_kernel(const gqa_row *rows, const uint8_t *fresh, uint32_t token_bytes) {
    const gqa_row R = rows[blockIdx.x];
    if (R.ring && R.pos + R.cap <= R.p_last) return;   /* a later row of this call takes the slot */
    const uint4 *src = reinterpret_cast<const uint4 *>(fresh + (uint64_t)blockIdx.x * token_bytes);
    uint4 *dst = reinterpret_cast<uint4 *>(R.cache + (uint64_t)(R.ring ? R.pos % R.cap : R.pos) * token_bytes);
    for (uint32_t i = threadIdx.x; i < token_bytes / 16u; i += blockDim.x) dst[i] = src[i];
}

/* ---- workspace geometry: ONE function sizes it and carves it ---------------------- */
/// The workspace carve (gqa_ws_layout); every region 256-byte aligned.
struct gqa_ws {
    /// Row descriptors (pulsar_gqa_launch points this at desc_dev instead).
    gqa_row       *rows;
    /// Tile descriptors (pulsar_gqa_launch points this at desc_dev instead).
    gqa_tile      *tiles;
    /// Prepped q, bf16 [n_rows][n_q][head_dim].
    __nv_bfloat16 *q;
    /// The call's KV rows, [n_rows][token_bytes].
    uint8_t       *fresh;
    /// Split-schedule accumulators, f32 [n_rows][n_q][max_chunks][head_dim] (empty at >= GQA_FOLD_ROWS rows).
    float         *part;
    /// Split-schedule (max, denominator), [n_rows][n_q][max_chunks].
    float2        *ml;
    /// pulsar_gqa_forward's own non-finite flag: the word it hands the launch when the caller gave none (O59).
    uint32_t      *nf;
    /// Chunks a row can span at the layout's context (gqa_max_chunks).
    uint32_t       max_chunks;
    /// Total bytes carved.
    uint64_t       bytes;
};

/// Whether this build's tensor-core arms widen their bf16 operands to fp16: device code compiled below sm_80 (the
/// host pass cannot ask PULSAR_SM80, which it always sees as 1; nvcc gives it the target list instead).  A build
/// whose targets nvcc does not name is taken to widen: the flag is then asked for and never written.
/// @return true when a launch's operands can leave the MMA's range (sm_75)
bool gqa_build_widens() {
#ifdef __CUDA_ARCH_LIST__
    const int archs[] = {__CUDA_ARCH_LIST__};
    for (const int a : archs)
        if (a < 800) return true;
    return false;
#else
    return true;
#endif
}

/// @param x a byte count.
/// @return x rounded up to a multiple of 256.
uint64_t gqa_up(uint64_t x) { return (x + 255u) & ~255ull; }

/// @param L the layer.
/// @param max_ctx a bound on the call's positions (all < max_ctx).
/// @return the most chunks one row's key range can touch.
uint32_t gqa_max_chunks(const pulsar_gqa_layer *L, uint32_t max_ctx) {
    const uint32_t full = (max_ctx + GQA_CHUNK - 1u) / GQA_CHUNK;
    if (L->span != PULSAR_GQA_WINDOW) return full;
    /* a block row also reads its run's later rows: at most GQA_CHUNK - 1 keys past it (gqa_plan bounds the run) */
    const uint32_t ahead = L->block ? GQA_CHUNK - 1u : 0u;
    return std::min(full, (L->window - 1u + ahead + GQA_CHUNK - 1u) / GQA_CHUNK + 1u);
}

/// @param L the layer.
/// @param max_ctx a bound on the call's positions (all < max_ctx).
/// @return the most chunks one TILE's key range can touch: the split schedule's chunk CTAs count from the tile's first
/// row's chunk (gqa_attn_kernel) up to its last row's, so a tile reaches its later rows' keys past one row's range.
uint32_t gqa_tile_chunks(const pulsar_gqa_layer *L, uint32_t max_ctx) {
    const uint32_t full = (max_ctx + GQA_CHUNK - 1u) / GQA_CHUNK;
    if (L->span != PULSAR_GQA_WINDOW) return full;
    /* keys past the first row's own position: a block row's run (gqa_max_chunks; every row of a run reads to its
     * end), else the tile's later rows -- 64 / G - 1 of them (L370: at a row's count, a 1024-key window's tile across
     * a chunk boundary had no CTA for its last chunk, and the combine folded a partial nothing had written) */
    const uint32_t reach = L->block ? GQA_CHUNK - 1u : GQA_MROWS / (L->n_q / L->n_kv) - 1u;
    return std::min(full, (L->window - 1u + reach + GQA_CHUNK - 1u) / GQA_CHUNK + 1u);
}

/// The ONE workspace sizer and carver.
/// @param base workspace base (256-byte aligned), or NULL to size only (pointers come back NULL).
/// @param L the layer.
/// @param n_rows the call's rows (partials are carved only below GQA_FOLD_ROWS).
/// @param max_ctx the context bound.
/// @return the carve.
gqa_ws gqa_ws_layout(uint8_t *base, const pulsar_gqa_layer *L, uint32_t n_rows, uint32_t max_ctx) {
    gqa_ws w{};
    w.max_chunks = gqa_max_chunks(L, max_ctx);
    const uint64_t tb = pulsar_gqa_kv_token_bytes(L);   /* the one authority for a token's K + V bytes (L335) */
    const uint64_t part_rows = n_rows < GQA_FOLD_ROWS ? n_rows : 0u;   /* partials: the split schedule only */
    uint64_t off = 0;
    auto take = [&](uint64_t bytes) { uint8_t *p = base ? base + off : nullptr; off += gqa_up(bytes); return p; };
    w.rows  = (gqa_row *)take((uint64_t)n_rows * sizeof(gqa_row));
    w.tiles = (gqa_tile *)take((uint64_t)n_rows * sizeof(gqa_tile));
    w.q     = (__nv_bfloat16 *)take((uint64_t)n_rows * L->n_q * L->head_dim * 2u);
    w.fresh = (uint8_t *)take((uint64_t)n_rows * tb);
    w.part  = (float *)take(part_rows * L->n_q * w.max_chunks * L->head_dim * sizeof(float));
    w.ml    = (float2 *)take(part_rows * L->n_q * w.max_chunks * sizeof(float2));
    w.nf    = (uint32_t *)take(sizeof(uint32_t));
    w.bytes = off;
    return w;
}

/// Names a refused contract on stderr.
/// @param why the reason.
/// @return -1.
int gqa_refuse(const char *why) {
    fprintf(stderr, "pulsar: gqa attention refused: %s\n", why);
    return -1;
}

/* The log-length temperature at `pos`, in the reference's f32 operations:
 * 1 + alpha * log(max(1, f32(pos + 1) / floor)); the log is taken in double and
 * rounded once (torch's f32 log is within an ulp of it); volatile keeps the
 * product and the sum two roundings. */
/// @param L the layer (temp_floor, temp_alpha).
/// @param pos the row's position.
/// @return the log-length temperature, 1 when the layer declares none.
float gqa_tau(const pulsar_gqa_layer *L, uint32_t pos) {
    if (L->temp_floor <= 0.f) return 1.0f;
    volatile float ratio = (float)(pos + 1u) / L->temp_floor;
    const float r = ratio < 1.0f ? 1.0f : (float)ratio;
    volatile float prod = L->temp_alpha * (float)log((double)r);
    return 1.0f + prod;
}

/// Announces each new launch shape once (rule 5).
pulsar_shape_once g_gqa_announce;

/// Launches a planned call's kernels on `st`: prep, attention (split or folded), combine (split only), commit (not for a block).
/// @tparam HD head_dim.
/// @param L the layer.
/// @param pa the prep arguments.
/// @param aa the attention arguments.
/// @param n_rows rows in the call.
/// @param n_tiles query tiles.
/// @param grid_chunks the split schedule's chunk CTAs per (KV head, tile); the decode arms' are a row's (ws.max_chunks).
/// @param fold the folded schedule.
/// @param decode the rows are decode rows (the wide heads' flash-decoding arm).
/// @param shape the flash-decoding CTA's rows and heads (gqa_dec_pick).
/// @param mma_dec the decode rows take the MMA arm (gqa_decode_mma_kernel) instead.
/// @param st the stream.
/// @param ws the workspace carve (rows, fresh).
/// @return 0, or -3 on a launch failure.
template <int HD>
int gqa_launch(const pulsar_gqa_layer *L, const gqa_prep_args &pa, const gqa_args &aa, uint32_t n_rows,
               uint32_t n_tiles, uint32_t grid_chunks, bool fold, bool decode, gqa_dec_shape shape, bool mma_dec,
               cudaStream_t st, const gqa_ws &ws) {
    const uint32_t jobs = L->n_q + 2u * L->n_kv;
    if (L->rope_inv_freq) {
        gqa_prep_table_args ta;
        static_cast<gqa_prep_args &>(ta) = pa;
        ta.rope_inv = L->rope_inv_freq;
        gqa_prep_kernel<HD, gqa_prep_table_args><<<dim3(n_rows, (jobs + 7u) / 8u), 256, 0, st>>>(ta);
    } else {
        gqa_prep_kernel<HD, gqa_prep_args><<<dim3(n_rows, (jobs + 7u) / 8u), 256, 0, st>>>(pa);
    }
    const bool bias = L->rel_extent > 0u;
    if constexpr (HD > 128) {   /* L338: the wide heads (no bias, gate or block: refused at the declaration) */
        const dim3 grid(fold ? 1u : grid_chunks, L->n_kv * (HD / GQA_WIDE_D), n_tiles);
        /* a decode CTA counts chunks from each row's own first (not a tile's): a row's most (L370) */
        const uint32_t dec_chunks = std::min(grid_chunks, ws.max_chunks);
        if (decode && !fold) {   /* L338: decode / verify rows (never a prompt's): flash-decoding, a CTA a row (the combine
                                  * below) -- a verify's rows and a plain decode's are then the same arm's bytes, so
                                  * speculation never changes the output */
            if (mma_dec) {   /* L338: a wide GQA's decode rows on the tensor cores (pulsar_gqa_launch's pick) */
                const uint32_t rb = shape.rb;   /* rows a CTA: gqa_dmma_rows */
                gqa_decode_mma_dispatch<HD>(rb, dim3(L->n_q / GQA_DMMA_NV, dec_chunks, (n_rows + rb - 1u) / rb), aa,
                                            n_rows, st);
            } else {
                const uint32_t rb = shape.rb, hb = shape.hb;   /* pulsar_gqa_launch's pick */
                const dim3 dgrid(L->n_q / hb, dec_chunks, (n_rows + rb - 1u) / rb);
                if (L->kv_fmt == PULSAR_GQA_KV_E4M3) gqa_decode_dispatch<HD, true>(rb, hb, dgrid, aa, n_rows, st);
                else                                 gqa_decode_dispatch<HD, false>(rb, hb, dgrid, aa, n_rows, st);
            }
        } else if (L->kv_fmt == PULSAR_GQA_KV_E4M3) {
            if (fold) gqa_attn_wide_kernel<HD, true, true><<<grid, 128, 0, st>>>(aa);
            else      gqa_attn_wide_kernel<HD, false, true><<<grid, 128, 0, st>>>(aa);
        } else {
            if (fold) gqa_attn_wide_kernel<HD, true, false><<<grid, 128, 0, st>>>(aa);
            else      gqa_attn_wide_kernel<HD, false, false><<<grid, 128, 0, st>>>(aa);
        }
    } else if (L->kv_fmt == PULSAR_GQA_KV_E4M3) {   /* L298: no relative bias with E4M3 rows (refused at the declaration) */
        if (fold) gqa_attn_kernel<HD, false, true, true><<<dim3(1, L->n_kv, n_tiles), 128, 0, st>>>(aa);
        else      gqa_attn_kernel<HD, false, false, true><<<dim3(grid_chunks, L->n_kv, n_tiles), 128, 0, st>>>(aa);
    } else if (fold) {
        const dim3 grid(1, L->n_kv, n_tiles);
        if (bias) gqa_attn_kernel<HD, true, true, false><<<grid, 128, 0, st>>>(aa);
        else      gqa_attn_kernel<HD, false, true, false><<<grid, 128, 0, st>>>(aa);
    } else {
        const dim3 grid(grid_chunks, L->n_kv, n_tiles);
        if (bias) gqa_attn_kernel<HD, true, false, false><<<grid, 128, 0, st>>>(aa);
        else      gqa_attn_kernel<HD, false, false, false><<<grid, 128, 0, st>>>(aa);
    }
    if (!fold) gqa_combine_kernel<HD><<<dim3(n_rows, L->n_q), HD, 0, st>>>(aa);
    if (!L->block) gqa_commit_kernel<<<n_rows, 128, 0, st>>>(ws.rows, ws.fresh, aa.token_bytes);   /* a block's rows are drafts */
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "pulsar: gqa attention launch failed: %s\n", cudaGetErrorString(e));
        return -3;
    }
    return 0;
}

/// @param in a strided input.
/// @param heads its head count.
/// @param width elements per head.
/// @return true when non-NULL with head and row strides that keep the heads from overlapping.
bool gqa_in_ok(const pulsar_gqa_in &in, uint32_t heads, uint32_t width) {
    return in.ptr && in.head_stride >= width && in.ld >= (uint64_t)(heads - 1u) * in.head_stride + width;
}

}  // namespace

uint64_t pulsar_gqa_kv_token_bytes(const pulsar_gqa_layer *layer) {
    return (uint64_t)layer->n_kv * 2u * layer->head_dim * (layer->kv_fmt == PULSAR_GQA_KV_E4M3 ? 1u : 2u);
}

uint64_t pulsar_gqa_workspace_bytes(const pulsar_gqa_layer *layer, uint32_t n_rows, uint32_t max_ctx) {
    /* Sized for EVERY call of at most n_rows rows: one below GQA_FOLD_ROWS takes the split schedule and
     * needs partials an n_rows >= GQA_FOLD_ROWS layout does not carve. */
    const uint64_t at_max = gqa_ws_layout(nullptr, layer, n_rows, max_ctx).bytes;
    if (n_rows < GQA_FOLD_ROWS) return at_max;
    const uint64_t split = gqa_ws_layout(nullptr, layer, GQA_FOLD_ROWS - 1u, max_ctx).bytes;
    return std::max(at_max, split);
}

uint64_t pulsar_gqa_desc_bytes(uint32_t n_rows) {
    return gqa_up((uint64_t)n_rows * sizeof(gqa_row)) + gqa_up((uint64_t)n_rows * sizeof(gqa_tile));
}

/* The layer's declaration, checked by a plan and a launch alike. */
/// @param L the layer's declaration.
/// @return 0 when every built tile covers it, else -1 (named on stderr).
static int gqa_layer_refusal(const pulsar_gqa_layer *L) {
    if (L->n_q == 0 || L->n_kv == 0 || L->n_q % L->n_kv) return gqa_refuse("n_q is not a positive multiple of n_kv");
    if (L->n_q / L->n_kv > GQA_MAX_GROUP) return gqa_refuse("the query group n_q / n_kv is wider than 64 (one CTA's MMA rows)");
    if (L->head_dim != 64u && L->head_dim != 128u && L->head_dim != 256u && L->head_dim != 512u)
        return gqa_refuse("head_dim has no built tile (64, 128, and the wide 256, 512)");
    if (L->head_dim > 128u && (L->rel_extent || L->out_gate || L->block))
        return gqa_refuse("the wide heads (256, 512) are built without a relative bias, an output gate or a block run");
    if (L->kv_fmt != PULSAR_GQA_KV_BF16 && L->kv_fmt != PULSAR_GQA_KV_E4M3)
        return gqa_refuse("the KV row format has no tile loader (BF16, E4M3)");
    if (L->kv_fmt == PULSAR_GQA_KV_E4M3 && L->rel_extent) return gqa_refuse("E4M3 KV rows with a relative bias have no built tile");
    if (L->span == PULSAR_GQA_FULL ? L->window != 0u : (L->span != PULSAR_GQA_WINDOW || L->window == 0u)) {
        return gqa_refuse("span / window disagree (FULL takes window 0, WINDOW a window >= 1)");
    }
    if (!(L->scale > 0.f) || !isfinite(L->scale)) return gqa_refuse("the softmax scale is not a finite positive number");
    if (L->qk_norm && (!L->q_norm_w || !L->k_norm_w || !(L->norm_eps >= 0.f))) {
        return gqa_refuse("qk_norm is declared but a norm weight or eps is missing");
    }
    if (L->rel_extent && (L->rel_dim != GQA_REL_DIM || !L->rel_proj)) {
        return gqa_refuse("the relative bias needs rel_dim 16 (one MMA k-step) and its projection table");
    }
    if (L->temp_floor < 0.f || (L->temp_floor > 0.f && !isfinite(L->temp_alpha))) {
        return gqa_refuse("the temperature floor is negative or its alpha is not finite");
    }
    if (L->rope_theta < 0.f || !isfinite(L->rope_theta)) return gqa_refuse("the rope base is negative or not finite");
    if (L->rope_inv_freq && !(L->rope_theta > 0.f)) return gqa_refuse("a rope frequency table with the rope off (rope_theta 0)");
    if (L->block && L->rel_extent) return gqa_refuse("a block (bidirectional run) with a relative bias has no built tile");
    return 0;
}

int pulsar_gqa_plan(const pulsar_gqa_layer *L, const pulsar_gqa_seq *seqs, uint32_t n_seqs,
                    const uint32_t *row_seq, const uint32_t *row_pos, uint32_t n_rows, uint32_t ctx_bound,
                    void *desc, uint64_t desc_cap, pulsar_gqa_call *out) {
    if (!L || !seqs || !row_seq || !row_pos || !out || n_rows == 0 || n_seqs == 0) {
        return gqa_refuse("a NULL argument or zero rows");
    }
    if (n_rows > 65535u) return gqa_refuse("more than 65535 rows in one call (the grid's tile dimension)");
    if (const int rc = gqa_layer_refusal(L)) return rc;
    if (!desc || desc_cap < pulsar_gqa_desc_bytes(n_rows)) {
        return gqa_refuse("the descriptor buffer is short of pulsar_gqa_desc_bytes");
    }
    const uint32_t G = L->n_q / L->n_kv;
    const uint64_t tb = pulsar_gqa_kv_token_bytes(L);

    /* rows -> descriptors; every sequence is ONE run of consecutive positions */
    std::vector<uint32_t> run_first(n_seqs, UINT32_MAX), run_last(n_seqs, UINT32_MAX);
    uint32_t max_ctx = 0;
    for (uint32_t r = 0; r < n_rows; r++) {
        const uint32_t sq = row_seq[r];
        if (sq >= n_seqs) return gqa_refuse("row_seq names no sequence");
        const pulsar_gqa_seq &S = seqs[sq];
        /* L321: a FULL block run from position 0 (an encoder's whole input) reads only its own fresh rows -- there are
         * no earlier keys -- and stores nothing, so it may name no pool at all (checked below: the run starts at 0) */
        const bool poolless = L->block && L->span == PULSAR_GQA_FULL && !S.kv;
        if (!poolless && (!S.kv || S.cap == 0 || S.bytes < (uint64_t)S.cap * tb || ((uintptr_t)S.kv & 15u))) {
            return gqa_refuse("a sequence's pool is missing, short, or not 16-byte aligned");
        }
        if (!poolless && L->span == PULSAR_GQA_FULL && row_pos[r] >= S.cap) return gqa_refuse("a row's position is past its pool's cap");
        if (L->span == PULSAR_GQA_WINDOW && S.cap < L->window) return gqa_refuse("a ring holds fewer slots than the window");
        if (row_pos[r] >= 0x7fffffffu) return gqa_refuse("a row position past 2^31");
        if (run_first[sq] == UINT32_MAX) {
            run_first[sq] = r;
        } else if (run_last[sq] != r - 1u || row_pos[r] != row_pos[r - 1u] + 1u) {
            return gqa_refuse("a sequence's rows are not one contiguous run of consecutive positions");
        }
        run_last[sq] = r;
        max_ctx = std::max(max_ctx, row_pos[r] + 1u);
    }
    if (ctx_bound && max_ctx > ctx_bound) return gqa_refuse("a row's position is at or past the call's ctx_bound");
    for (uint32_t sq = 0; L->block && sq < n_seqs; sq++) {
        if (run_first[sq] == UINT32_MAX) continue;
        /* a WINDOW block's chunk bound counts at most GQA_CHUNK - 1 keys past a row (gqa_max_chunks); a FULL one's is the
         * context's, so a FULL run (an encoder's input, L321) may be as long as the call */
        if (L->span == PULSAR_GQA_WINDOW && run_last[sq] - run_first[sq] >= GQA_CHUNK)
            return gqa_refuse("a windowed block run longer than 256 rows (a block row reads at most 255 keys past it)");
        if (!seqs[sq].kv && row_pos[run_first[sq]] != 0)
            return gqa_refuse("a block run with no pool must start at position 0 (it has no earlier keys to read)");
    }

    gqa_row *h_rows = (gqa_row *)desc;
    gqa_tile *h_tiles = (gqa_tile *)((uint8_t *)desc + gqa_up((uint64_t)n_rows * sizeof(gqa_row)));
    for (uint32_t r = 0; r < n_rows; r++) {
        const uint32_t sq = row_seq[r], p = row_pos[r];
        gqa_row &d = h_rows[r];
        d.cache  = (uint8_t *)seqs[sq].kv;
        d.pos    = p;
        d.lo     = L->span == PULSAR_GQA_WINDOW && p + 1u > L->window ? p + 1u - L->window : 0u;
        d.p0     = row_pos[run_first[sq]];
        d.fresh0 = run_first[sq];
        d.p_last = row_pos[run_last[sq]];
        d.cap    = seqs[sq].cap;
        d.ring   = L->span == PULSAR_GQA_WINDOW ? 1u : 0u;
        d.chunk0 = d.lo / GQA_CHUNK;
        d.tau    = gqa_tau(L, p);
        d.hi     = L->block ? row_pos[run_last[sq]] : p;
    }
    /* tiles: runs of up to 64 / G consecutive rows of one sequence */
    const uint32_t qt = GQA_MROWS / G;
    uint32_t n_tiles = 0, grid_chunks = 1;
    for (uint32_t r = 0; r < n_rows;) {
        uint32_t n = 1;
        while (n < qt && r + n < n_rows && row_seq[r + n] == row_seq[r]) n++;
        h_tiles[n_tiles++] = {r, n};
        grid_chunks = std::max(grid_chunks, h_rows[r + n - 1u].hi / GQA_CHUNK - h_rows[r].chunk0 + 1u);
        r += n;
    }
    /* A bounded call's split grid is the bound's -- the most chunks a TILE can touch (gqa_tile_chunks), which is what
     * the loop above counts for the tiles at hand.  A CTA whose chunk is past its tile's last row exits before it
     * reads anything, and the combine folds each row's own chunks: the launches then depend on (n_rows, n_tiles, the
     * bound) and never on the positions -- what a captured step replays.  The bytes are the unbounded call's. */
    if (ctx_bound) grid_chunks = gqa_tile_chunks(L, ctx_bound);
    out->n_rows = n_rows;
    out->n_tiles = n_tiles;
    out->grid_chunks = grid_chunks;
    out->ctx_bound = ctx_bound ? ctx_bound : max_ctx;
    out->fold = n_rows >= GQA_FOLD_ROWS ? 1 : 0;
    return 0;
}

int pulsar_gqa_launch(const pulsar_gqa_layer *L, const pulsar_gqa_call *call, const void *desc_dev,
                      const pulsar_gqa_io *io, cudaStream_t st) {
    if (!L || !call || !desc_dev || !io || call->n_rows == 0 || call->n_tiles == 0) {
        return gqa_refuse("a NULL argument or an empty plan");
    }
    if (const int rc = gqa_layer_refusal(L)) return rc;
    const uint32_t G = L->n_q / L->n_kv, HD = L->head_dim, n_rows = call->n_rows;
    if (!gqa_in_ok(io->q, L->n_q, HD) || !gqa_in_ok(io->k, L->n_kv, HD) || !gqa_in_ok(io->v, L->n_kv, HD)) {
        return gqa_refuse("a q / k / v input is missing or its strides overlap its heads");
    }
    if (L->rel_extent && (!gqa_in_ok(io->r, L->n_q, L->rel_dim) || ((uintptr_t)io->r.ptr & 3u) || (io->r.ld & 1u) ||
                          (io->r.head_stride & 1u))) {
        return gqa_refuse("the relative bias's r input is missing, overlapping, or not bf16-pair aligned");
    }
    if (L->out_gate && !gqa_in_ok(io->gate, L->n_q, HD)) return gqa_refuse("out_gate is declared but its gate input is missing");
    if (!io->out_bf16 || ((uintptr_t)io->out_bf16 & 3u) || (io->ld_out & 1u) || io->ld_out < (uint64_t)L->n_q * HD) {
        return gqa_refuse("out_bf16 is missing, not bf16-pair aligned, or narrower than n_q * head_dim");
    }
    if (((uintptr_t)io->workspace & 255u) != 0 || ((uintptr_t)desc_dev & 255u) != 0) {
        return gqa_refuse("the workspace or the descriptors are not 256-byte aligned");
    }
    const uint64_t tb = pulsar_gqa_kv_token_bytes(L);
    const gqa_ws need = gqa_ws_layout(nullptr, L, n_rows, call->ctx_bound);
    if (!io->workspace || io->workspace_bytes < need.bytes) {
        fprintf(stderr, "pulsar: gqa attention refused: workspace %llu B < %llu B for %u rows at ctx %u\n",
                (unsigned long long)io->workspace_bytes, (unsigned long long)need.bytes, n_rows, call->ctx_bound);
        return -1;
    }
    gqa_ws ws = gqa_ws_layout((uint8_t *)io->workspace, L, n_rows, call->ctx_bound);
    ws.rows = (gqa_row *)desc_dev;
    ws.tiles = (gqa_tile *)((const uint8_t *)desc_dev + gqa_up((uint64_t)n_rows * sizeof(gqa_row)));
    const bool fold = call->fold != 0;

    const uint64_t key = pulsar_shape_key(((uint64_t)n_rows << 1) | (fold ? 1u : 0u),
                                          ((uint64_t)L->n_q << 24) ^ ((uint64_t)L->n_kv << 16) ^ ((uint64_t)HD << 4) ^
                                          ((uint64_t)L->span << 2) ^ (L->rel_extent ? 2u : 0u) ^ (L->temp_floor > 0.f ? 1u : 0u) ^
                                          (L->rope_theta > 0.f ? 0x100u : 0u) ^ (L->block ? 0x200u : 0u) ^
                                          (L->kv_fmt == PULSAR_GQA_KV_E4M3 ? 0x400u : 0u) ^ (L->rope_inv_freq ? 0x800u : 0u));
    if (pulsar_shape_once_first(&g_gqa_announce, key, "gqa attention")) {
        fprintf(stderr, "pulsar: gqa attention: %uq/%ukv/%u, KV %s, %s %u, rel bias %s (extent %u), temperature %s, "
                        "qk norm %s, gate %s, rope %s, %s: %u rows in %u tiles, %s\n",
                L->n_q, L->n_kv, HD, L->kv_fmt == PULSAR_GQA_KV_E4M3 ? "E4M3" : "BF16",
                L->span == PULSAR_GQA_WINDOW ? "window" : "full", L->window,
                L->rel_extent ? "on" : "off", L->rel_extent, L->temp_floor > 0.f ? "on" : "off",
                L->qk_norm ? "on" : "off", L->out_gate ? "on" : "off", L->rope_theta > 0.f ? (L->rope_inv_freq ? "neox (frequency table)" : "neox") : "off",
                L->block ? "bidirectional runs, not stored" : "causal", n_rows, call->n_tiles,
                fold ? "chunks folded in the CTA" : "chunks in parallel + combine");
    }

    gqa_prep_args pa{};
    pa.rows = ws.rows;
    pa.qin = (const __nv_bfloat16 *)io->q.ptr; pa.ld_q = io->q.ld; pa.hs_q = io->q.head_stride;
    pa.kin = (const __nv_bfloat16 *)io->k.ptr; pa.ld_k = io->k.ld; pa.hs_k = io->k.head_stride;
    pa.vin = (const __nv_bfloat16 *)io->v.ptr; pa.ld_v = io->v.ld; pa.hs_v = io->v.head_stride;
    pa.qw = L->qk_norm ? L->q_norm_w : nullptr;
    pa.kw = L->qk_norm ? L->k_norm_w : nullptr;
    pa.eps = L->norm_eps;
    pa.n_q = L->n_q; pa.n_kv = L->n_kv; pa.token_bytes = (uint32_t)tb;
    pa.rope_theta = L->rope_theta;
    pa.e4m3 = L->kv_fmt == PULSAR_GQA_KV_E4M3 ? 1u : 0u;
    pa.q = ws.q; pa.fresh = ws.fresh;

    gqa_args aa{};
    aa.rows = ws.rows; aa.tiles = ws.tiles; aa.q = ws.q; aa.fresh = ws.fresh;
    aa.r = L->rel_extent ? (const __nv_bfloat16 *)io->r.ptr : nullptr;
    aa.ld_r = io->r.ld; aa.r_hs = io->r.head_stride;
    aa.proj = (const __nv_bfloat16 *)L->rel_proj; aa.extent = L->rel_extent;
    aa.scale = L->scale;
    aa.n_q = L->n_q; aa.n_kv = L->n_kv; aa.G = G; aa.token_bytes = (uint32_t)tb; aa.max_chunks = ws.max_chunks;
    aa.part = ws.part; aa.ml = ws.ml;
    aa.gate = L->out_gate ? (const __nv_bfloat16 *)io->gate.ptr : nullptr;
    aa.ld_gate = io->gate.ld; aa.gate_hs = io->gate.head_stride;
    aa.out = (__nv_bfloat16 *)io->out_bf16; aa.ld_out = io->ld_out;
    aa.tap = io->tap_out_f32;
    /* O59: where the arms widen bf16 to fp16 (sm_75) an operand can leave the MMA's range, and the kernels say so in
     * the step's flag: a launch with nowhere to say it does not run */
    aa.nf = io->nf_flag; aa.nf_code = io->nf_code;
    if (gqa_build_widens() && (!aa.nf || aa.nf_code == 0u)) {
        return gqa_refuse("no non-finite flag (pulsar_gqa_io::nf_flag with a nonzero nf_code): this build's arms widen "
                          "bf16 operands to fp16 and report one outside fp16's range there");
    }

    /* L338: rows a flash-decoding CTA attends (the caller's: a verify's rows share one sequence's keys) */
    if (io->decode_group != 0 && io->decode_group != 1 && io->decode_group != 2 && io->decode_group != 4)
        return gqa_refuse("a decode group of other than 0, 1, 2 or 4 rows");
    /* the flash-decoding CTA's shape (decode rows of a wide-head layer): a function of the layer and n_rows (what a
     * captured step's key already holds) */
    gqa_dec_shape shape = gqa_dec_pick(G, L->n_q, std::min(call->grid_chunks, ws.max_chunks), n_rows,
                                             io->decode_group <= 1 ? 1u : (uint32_t)io->decode_group);
    if (G % shape.hb != 0u) return gqa_refuse("a forced decode shape whose heads do not divide the layer's G");
    /* L338: a wide GQA's decode rows (G a multiple of 8, BF16 rows) on the tensor cores -- a function of the layer
     * alone: a row's arm (its bytes) may not depend on its call's rows or bound, or a verify's rows and a plain
     * decode's would differ.  Measured on the RTX 5000 (gqa_attn_bench GQA_BENCH_ARM, the global layer, one row):
     * 86 vs flash-decoding's 87 us at 512 keys, 91 vs 135 at 2K, 165 vs 337 at 8K, 451 vs 984 at 32K.  An
     * instrument may force either arm. */
    const bool mma_ok = HD % 64u == 0u && HD <= 512u && G % GQA_DMMA_NV == 0u && L->kv_fmt == PULSAR_GQA_KV_BF16;
    if (g_dec_arm_force == 2 && !mma_ok) return gqa_refuse("the MMA decode arm forced on a layer it does not run");
    const bool mma_dec = io->decode_rows != 0 && call->fold == 0 && mma_ok && g_dec_arm_force != 1;
    /* its rows a CTA (the caller's most, by the row count: a (row, head)'s bytes are the same at any) */
    const uint32_t dmma_rb = io->decode_group >= 4 && n_rows >= 3u ? 4u : io->decode_group >= 2 && n_rows >= 2u ? 2u : 1u;
    if (mma_dec) shape = {dmma_rb, GQA_DMMA_NV};
    switch (HD) {
    case 512u: return gqa_launch<512>(L, pa, aa, n_rows, call->n_tiles, call->grid_chunks, fold, io->decode_rows != 0, shape, mma_dec, st, ws);
    case 256u: return gqa_launch<256>(L, pa, aa, n_rows, call->n_tiles, call->grid_chunks, fold, io->decode_rows != 0, shape, mma_dec, st, ws);
    case 128u: return gqa_launch<128>(L, pa, aa, n_rows, call->n_tiles, call->grid_chunks, fold, io->decode_rows != 0, shape, mma_dec, st, ws);
    default:   return gqa_launch<64>(L, pa, aa, n_rows, call->n_tiles, call->grid_chunks, fold, io->decode_rows != 0, shape, mma_dec, st, ws);
    }
}

int pulsar_gqa_forward(const pulsar_gqa_layer *L, const pulsar_gqa_seq *seqs, uint32_t n_seqs,
                       const uint32_t *row_seq, const uint32_t *row_pos, uint32_t n_rows,
                       const pulsar_gqa_io *io, cudaStream_t st) {
    if (!io || n_rows == 0 || n_rows > 65535u) return gqa_refuse("a NULL argument or a row count outside 1 .. 65535");
    /* the plan staged on the host in the workspace's own layout (its first two regions are the rows and the tiles),
     * one upload, then the launch from it */
    std::vector<uint8_t> stage(pulsar_gqa_desc_bytes(n_rows));
    pulsar_gqa_call call{};
    if (const int rc = pulsar_gqa_plan(L, seqs, n_seqs, row_seq, row_pos, n_rows, 0u, stage.data(), stage.size(), &call)) {
        return rc;
    }
    const uint64_t need = gqa_ws_layout(nullptr, L, n_rows, call.ctx_bound).bytes;
    if (!io->workspace || ((uintptr_t)io->workspace & 255u) != 0 || io->workspace_bytes < need) {
        fprintf(stderr, "pulsar: gqa attention refused: workspace %llu B (want %llu B, 256-byte aligned) for %u rows "
                        "at ctx %u\n", (unsigned long long)io->workspace_bytes, (unsigned long long)need, n_rows,
                call.ctx_bound);
        return -1;
    }
    if (cudaMemcpyAsync(io->workspace, stage.data(), stage.size(), cudaMemcpyHostToDevice, st) != cudaSuccess) {
        fprintf(stderr, "pulsar: gqa attention: descriptor upload failed\n");
        return -3;
    }
    /* O59: a caller with no step flag of its own (the one-call form: the encoders, Parse) gets the call's verdict
     * here -- the flag is a word of the workspace, zeroed before the launch and read back after it */
    if (io->nf_flag || !gqa_build_widens()) return pulsar_gqa_launch(L, &call, io->workspace, io, st);
    pulsar_gqa_io own = *io;
    own.nf_flag = gqa_ws_layout((uint8_t *)io->workspace, L, n_rows, call.ctx_bound).nf;
    own.nf_code = 1u;
    if (cudaMemsetAsync(own.nf_flag, 0, sizeof(uint32_t), st) != cudaSuccess) {
        fprintf(stderr, "pulsar: gqa attention: the non-finite flag's reset failed\n");
        return -3;
    }
    if (const int rc = pulsar_gqa_launch(L, &call, io->workspace, &own, st)) return rc;
    uint32_t nf = 0u;
    if (cudaMemcpyAsync(&nf, own.nf_flag, sizeof(nf), cudaMemcpyDeviceToHost, st) != cudaSuccess ||
        cudaStreamSynchronize(st) != cudaSuccess) {
        fprintf(stderr, "pulsar: gqa attention: the non-finite flag's readback failed\n");
        return -3;
    }
    if (nf) {
        fprintf(stderr, "pulsar: gqa attention refused: a q / k / v operand (or a relative bias's r / table) of this "
                        "%uq/%ukv/%u call is outside fp16's range (|x| > 65504, or not finite); this build's arms "
                        "widen bf16 to fp16 for the MMA, so its output is not the attention's (sm_80+ carries bf16 "
                        "exactly)\n", L->n_q, L->n_kv, L->head_dim);
        return -4;
    }
    return 0;
}

int pulsar_gqa_widens(void) { return gqa_build_widens() ? 1 : 0; }

/* ---- the KV rows alone (L298): gqa_prep's K / V jobs and gqa_commit, nothing between --------------------------- */

uint64_t pulsar_gqa_store_workspace_bytes(const pulsar_gqa_layer *layer, uint32_t n_rows) {
    return pulsar_gqa_desc_bytes(n_rows) + gqa_up((uint64_t)n_rows * pulsar_gqa_kv_token_bytes(layer));
}

int pulsar_gqa_store(const pulsar_gqa_layer *L, const pulsar_gqa_seq *seqs, uint32_t n_seqs, const uint32_t *row_seq,
                     const uint32_t *row_pos, uint32_t n_rows, const pulsar_gqa_io *io, cudaStream_t st) {
    if (!L || !io || n_rows == 0 || n_rows > 65535u) return gqa_refuse("a NULL argument or a row count outside 1 .. 65535");
    if (L->block) return gqa_refuse("a store of a block's rows (a block's rows are drafts, never stored)");
    if (!gqa_in_ok(io->k, L->n_kv, L->head_dim) || !gqa_in_ok(io->v, L->n_kv, L->head_dim)) {
        return gqa_refuse("a store's k / v input is missing or its strides overlap its heads");
    }
    const uint64_t desc = pulsar_gqa_desc_bytes(n_rows), tb = pulsar_gqa_kv_token_bytes(L);
    if (!io->workspace || ((uintptr_t)io->workspace & 255u) != 0 ||
        io->workspace_bytes < pulsar_gqa_store_workspace_bytes(L, n_rows)) {
        return gqa_refuse("the store's workspace is missing, short or not 256-byte aligned");
    }
    /* the forward's plan (its contract and refusals) into the workspace's descriptor region */
    std::vector<uint8_t> stage(desc);
    pulsar_gqa_call call{};
    if (const int rc = pulsar_gqa_plan(L, seqs, n_seqs, row_seq, row_pos, n_rows, 0u, stage.data(), stage.size(), &call)) {
        return rc;
    }
    if (cudaMemcpyAsync(io->workspace, stage.data(), stage.size(), cudaMemcpyHostToDevice, st) != cudaSuccess) {
        fprintf(stderr, "pulsar: gqa store: descriptor upload failed\n");
        return -3;
    }
    const gqa_row *rows = (const gqa_row *)io->workspace;
    uint8_t *fresh = (uint8_t *)io->workspace + desc;
    gqa_prep_args pa{};
    pa.rows = rows;
    pa.kin = (const __nv_bfloat16 *)io->k.ptr; pa.ld_k = io->k.ld; pa.hs_k = io->k.head_stride;
    pa.vin = (const __nv_bfloat16 *)io->v.ptr; pa.ld_v = io->v.ld; pa.hs_v = io->v.head_stride;
    pa.kw = L->qk_norm ? L->k_norm_w : nullptr;
    pa.eps = L->norm_eps;
    pa.n_q = 0;   /* no query jobs: job g < n_kv is K head g, then the V heads */
    pa.n_kv = L->n_kv; pa.token_bytes = (uint32_t)tb;
    pa.rope_theta = L->rope_theta;
    pa.e4m3 = L->kv_fmt == PULSAR_GQA_KV_E4M3 ? 1u : 0u;
    pa.fresh = fresh;
    const dim3 grid(n_rows, (2u * L->n_kv + 7u) / 8u);
    gqa_prep_table_args ta;
    static_cast<gqa_prep_args &>(ta) = pa;
    ta.rope_inv = L->rope_inv_freq;
    auto prep = [&](auto hd) {
        constexpr int HDc = decltype(hd)::value;
        if (L->rope_inv_freq) gqa_prep_kernel<HDc, gqa_prep_table_args><<<grid, 256, 0, st>>>(ta);
        else                  gqa_prep_kernel<HDc, gqa_prep_args><<<grid, 256, 0, st>>>(pa);
    };
    switch (L->head_dim) {
    case 512u: prep(std::integral_constant<int, 512>{}); break;
    case 256u: prep(std::integral_constant<int, 256>{}); break;
    case 128u: prep(std::integral_constant<int, 128>{}); break;
    default:   prep(std::integral_constant<int, 64>{}); break;
    }
    gqa_commit_kernel<<<n_rows, 128, 0, st>>>(rows, fresh, (uint32_t)tb);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "pulsar: gqa store launch failed: %s\n", cudaGetErrorString(e));
        return -3;
    }
    return 0;
}

int pulsar_gqa_decode_force(int rows, int heads) {
    if (rows == 0 && heads == 0) {
        g_dec_force = {0u, 0u};
        return 0;
    }
    const bool ok = (rows == 1 && (heads == 1 || heads == 2 || heads == 4)) || (rows == 2 && heads == 2);
    if (!ok) {
        fprintf(stderr, "pulsar: gqa decode force: %d rows x %d heads is not an instantiated shape (1 x 1, 1 x 2, 1 x 4, "
                        "2 x 2) -- refusing\n", rows, heads);
        return -1;
    }
    g_dec_force = {(uint32_t)rows, (uint32_t)heads};
    return 0;
}

int pulsar_gqa_decode_arm_force(int arm) {
    if (arm < 0 || arm > 2) {
        fprintf(stderr, "pulsar: gqa decode arm force: %d is not 0 (the pick), 1 (flash-decoding) or 2 (MMA) -- refusing\n",
                arm);
        return -1;
    }
    g_dec_arm_force = arm;
    return 0;
}

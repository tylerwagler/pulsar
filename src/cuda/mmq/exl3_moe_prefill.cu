// SPDX-License-Identifier: MIT
/** @file
 *  The EXL3 routed-expert projections on fp16 tensor cores: the prompt-chunk grouped GEMM and the decode split-K GEMM. */
// The EXL3 routed-expert projections on fp16 tensor cores, every family: the arm for PROMPT CHUNKS
// (L251 Qwen; L287 DeepSeek) and, below it, the arm for DECODE rows (task #46).  Both run over the
// expert-sorted schedule (mm_ids_helper's ids_src1 / ids_dst / expert_bounds) and write the UNROTATED
// z the fold and the sum consume.  Shapes are the call's (M, K, experts): nothing here is a family's.
// (Until task #46 decode rows took an f32 CUDA-core GEMV that decoded an expert's trellis once per
// assignment; it was 1.0-1.6x slower than the decode GEMM at every width it served, and is deleted.)
//
//   PREP  once per (assignment, 128-block): the row in the format its PRODUCER emitted (rule 3) --
//         bf16 rows (Qwen) or the E4M3 slot with its per-32 ue8m0 scales (DeepSeek; e4m3 x 2^e is
//         exact in f32) -- times the expert's suh and rotated by H128 when the projection rotates its
//         input (gate / up, fused or split; the down's input arrives rotated from the fold), a
//         power-of-two prescale per block, rounded to ONE fp16 plane: 11 significant bits relative
//         to the block's largest value -- 8x finer than a bf16 row, and EXACT for an unrotated E4M3
//         value (4 significant bits), so DeepSeek's down reads its slot's values bit for bit.
//         (exllamav3's and TensorFold's operand is the same one fp16 plane; a second "lo" plane
//         doubled the MMAs for precision the input does not have.)
//   GEMM  one CTA per (128-output tile, <= 64 consecutive assignments of ONE expert; the plan
//         kernel turns expert_bounds into that list): per 128-k block the A tiles go to shared
//         memory by cp.async while each of the 8 warps decodes its n16 tile's 8 k-tiles of the
//         expert's trellis ONCE into mma B registers (mul1_pair, exllamav3's codebook bit for bit),
//         then applies them to the item's live m16 tiles only;
//         then one MMA per n8 tile into a fresh f32 fragment, added into the accumulator
//         scaled by the (row, block) inverse prescale -- a fresh fragment because the tensor
//         core's own accumulate is not IEEE f32.
//   The trellis is decoded straight into the mma B fragments (TensorFold's experts design, "D2R"),
//   never to memory: decoding each routed expert ONCE to fp16 and running a cuBLASLt GEMM per expert
//   (the dense prefill arm's shape) writes and re-reads 2 bytes per weight against the trellis's 3/8,
//   and measured 4.0x slower on V4's gate projection at a 4096-token chunk (96 rows per expert) and
//   5.6x on V4.1's (64 rows per expert) -- tests/exl3_moe_prefill_bench, L287.
//
// A row's arithmetic is a function of its own operand row and its expert's weights alone (its own
// prescales, the fixed block / k-tile order, a fresh fragment per k-tile), so its bits never depend
// on its chunk or its batchmates (exl3_moe_prefill_gate proves it).  The first version staged A
// INSIDE the GEMM, so every assignment's row was gathered, rotated and split once per 64-output tile
// (20x for the fused gate_up) with no load pipelining: 11 TFLOP/s.  A prefilled assignment and a
// decoded one agree to rounding, not to the bit (Tyler 2026-09-29, "Don't use a decode kernel for
// prefill"); graded by exl3_moe_prefill_gate, qwen_s4_gate D and the reference gates.
//
// The DECODE GEMM (task #46) shares the PREP and the per-k-tile arithmetic (a fresh fragment per k-tile,
// scaled by the row's inverse prescale) but splits K: decode rows meet an expert a handful of times, so
// the grouped GEMM's serial-K CTAs (one chain over all of K per output) left most of GB10 idle at 1-8
// rows (Qwen's 640-wide gate: 5 output tiles per expert).  A CTA there is (64 outputs, a 16-row m16
// tile of one expert, one of SK K-splits), four warps over the split's k-range, summed in warp order;
// the splits are added in split order by the fold / sum.  SK is a function of K alone.  The tiles are
// the PREP's plan (L297): every expert's assignments cut into 16-row tiles, so an R-row step streams each
// routed expert's weights ONCE for all of its rows -- the bytes are the union of the rows' experts.  Each
// tile's words load coalesced (lane l word l) and a lane takes its 64-bit window from its neighbours by
// shuffle -- at K <= 4 one window holds all eight of a lane's B-fragment states (run8, TensorFold's load).

#include "exl3_moe_prefill.cuh"
#include "ds4_exl3_dev.cuh"
#include "cuda/pulsar_cuda_mx.cuh"
#include "engine/exl3_trellis.h"
#include "cuda/pulsar_cuda_arch.cuh"

#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

constexpr int kBM      = 64;          ///< assignments per CTA: four m16 tiles
constexpr int kBN      = 128;         ///< outputs per CTA: eight n16 tiles
constexpr int kThreads = 256;         ///< 8 warps: one n16 tile each
constexpr int kApad    = 128 + 8;     ///< halves per staged A row: 16-byte rows, conflict-free ldmatrix
constexpr int kHiExp   = 14;          ///< the prescale puts a block's largest |value| just under 2^14
constexpr int kPlanCap = 1 << 15;     ///< work items one call may plan (8192 rows x 10 slots / 64 + 1024 << this)
constexpr int kMaxE    = 1024;        ///< experts the one-block plan kernel scans

/* The tensor-core arms (prefill and decode) read exactly the rates the routed projection kinds admit
 * (exl3_arm_has_rate is the one table): a projection never meets a rate no arm serves, and the arms
 * instantiate nothing no kind admits. */
/// Whether EXL3_ARM_MOE_PREFILL admits exactly the rates the routed kinds (DOWN, PAIR, GATE_UP_FUSED) admit,
/// for every rate from k2 to 16 (a compile-time recursion).
/// @param k2  the first rate checked
/// @return    true when the tables agree
constexpr bool covers_routed_rates(int k2 = 0) {
    return k2 > 16 || (exl3_arm_has_rate(EXL3_ARM_MOE_PREFILL, k2) ==
                           (exl3_arm_has_rate(EXL3_ARM_DOWN, k2) || exl3_arm_has_rate(EXL3_ARM_PAIR, k2) ||
                            exl3_arm_has_rate(EXL3_ARM_GATE_UP_FUSED, k2)) &&
                       covers_routed_rates(k2 + 1));
}
static_assert(covers_routed_rates(), "EXL3_ARM_MOE_PREFILL's rates must be the routed projection kinds' rates");

/// The prefill GEMM's dynamic shared memory: the double-buffered A tiles and their inverse prescales (~35 KB).
struct Smem {
    __half a_hi[2][kBM][kApad];       ///< A of block b in buffer b & 1: block b + 1 loads while b computes
    /// The staged block's inverse prescales per assignment row, double-buffered like a_hi.
    float  inv[2][kBM];
};

/// 2^e as an f32 built from its exponent bits.
/// @param e  the exponent, in [-126, 127]
/// @return   2^e
__device__ __forceinline__ float pow2f(int e) { return __int_as_float((127 + e) << 23); }   /* e in [-126, 127] */

/// d += A * B on one mma.sync m16n8k16, fp16 operands, f32 accumulate; traps below sm_80 (the launchers refuse first).
/// @param d   the f32 C/D fragment, accumulated in place
/// @param a   the A fragment (four fp16x2 registers)
/// @param b0  B fragment register 0
/// @param b1  B fragment register 1
__device__ __forceinline__ void mma_16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if PULSAR_SM80
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}

/// Four 8x8 b16 matrices from shared memory into an MMA A fragment (ldmatrix .x4).
/// @param a  out: the four fragment registers
/// @param p  this lane's 16-byte row address in shared memory
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&a)[4], const __half *p) {
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(s));
}

/// 16 bytes global -> shared, asynchronously (cp.async.cg); traps below sm_80.
/// @param smem       16-byte aligned shared destination
/// @param gmem       16-byte aligned global source
/// @param src_bytes  bytes read (16, or 0 to zero-fill)
__device__ __forceinline__ void cp_async16(void *smem, const void *gmem, int src_bytes) {
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
#if PULSAR_SM80
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(src_bytes));
#else
    __trap();
#endif
}

/* The work list: expert e's assignments [bounds[e], bounds[e+1]) cut into
 * blocks of kBM, in expert order.  One block of E threads, an inclusive
 * Hillis-Steele scan over the per-expert block counts.  `cap` is the list's
 * length (the launch sizes it to n_assign / kBM + E + 1, which no routing exceeds). */
/// The prefill GEMM's work list.  Grid 1, block the next power of two >= max(E, 32) threads (<= kMaxE).
/// @param bounds  expert_bounds [E + 1], ascending
/// @param E       experts
/// @param work    out: items (expert, first assignment), in expert order
/// @param n_work  out: the item count, clamped to cap
/// @param cap     the list's length
__global__ void exl3_moe_plan_kernel(const int32_t *__restrict__ bounds, int E, int2 *__restrict__ work,
                                     int *__restrict__ n_work, int cap) {
    __shared__ int s[kMaxE];
    const int e = threadIdx.x;
    int mb = 0;
    if (e < E) mb = (bounds[e + 1] - bounds[e] + kBM - 1) / kBM;
    s[e] = mb;
    __syncthreads();
    for (int off = 1; off < (int)blockDim.x; off <<= 1) {
        const int v = e >= off ? s[e - off] : 0;
        __syncthreads();
        s[e] += v;
        __syncthreads();
    }
    const int start = s[e] - mb;
    for (int j = 0; j < mb; ++j)
        if (start + j < cap) work[start + j] = make_int2(e, bounds[e] + j * kBM);
    if (e == (int)blockDim.x - 1) *n_work = s[e] < cap ? s[e] : cap;
}


/* One projection's operand: the expert table whose suh rotates the input (ROT), the fp16 plane [n_assign][K] and
 * the inverse prescales [n_assign][K / 128] the PREP writes for it. */
/// One projection's operand as the PREP writes it.
struct prep_out {
    /// The projection's expert table (exl3_expert_table()'s pairs): its suh rotates the input when ROT.
    const void *table;
    /// The fp16 operand plane [n_assign][K].
    __half *phi;
    /// Inverse prescales f32 [n_assign][K / 128].
    float *pinv;
};

constexpr int kDecRows = 16;   ///< the decode GEMM's tile: up to 16 consecutive assignments of ONE expert

/* The decode GEMM's tile list (L297): tile t = (expert, its first assignment), every expert's assignments cut into
 * kDecRows-row tiles, in expert order -- the tiles the decode grid used to find by a binary search per CTA, its
 * idle CTAs included.  NULL tiles: no plan (the prefill arm plans its own work list). */
/// The decode GEMM's planned tile list (device), or both NULL for none.
struct dec_plan {
    int2 *tiles;    ///< [<= n_assign]: a tile holds >= 1 assignment
    /// The tile count, written by the plan.
    int  *n_tiles;
};

/* The plan, by one 256-thread CTA: thread t counts the tiles of experts [t per, (t + 1) per), an inclusive scan
 * over the counts, then each thread writes its experts' tiles from its offset. */
/// Writes the decode tile list; called by one whole 256-thread CTA (the prep grid's last).
/// @param bounds  expert_bounds [E + 1]
/// @param E       experts
/// @param p       where the tiles and their count go
__device__ void plan_decode_tiles(const int32_t *__restrict__ bounds, int E, const dec_plan &p) {
    __shared__ int s[256];
    const int t = threadIdx.x, per = (E + 255) / 256, e0 = min(E, t * per), e1 = min(E, e0 + per);
    int n = 0;
    for (int e = e0; e < e1; ++e) n += (bounds[e + 1] - bounds[e] + kDecRows - 1) / kDecRows;
    s[t] = n;
    __syncthreads();
    for (int off = 1; off < 256; off <<= 1) {
        const int v = t >= off ? s[t - off] : 0;
        __syncthreads();
        s[t] += v;
        __syncthreads();
    }
    int at = s[t] - n;
    for (int e = e0; e < e1; ++e)
        for (int a = bounds[e]; a < bounds[e + 1]; a += kDecRows) p.tiles[at++] = make_int2(e, a);
    if (t == 255) *p.n_tiles = s[255];
}

/* PREP: one warp per (assignment, 128-block); lane l owns k = 4l..4l+3 of the block.  FMT is the
 * producer's activation format (exl3_moe_act_format): the row is read as given, never re-encoded.
 * blockIdx.y picks the projection (a decode gate + up pair preps both in one launch).  With a plan (a decode
 * launch), the grid's last CTA of projection 0 writes the decode GEMM's tile list instead. */
/// PREP (launch_prep).  Grid (ceil(n_assign * K/128 / 8) [+ 1 with a plan], n_proj), 256 threads.
/// @param o0        projection 0's operand
/// @param o1        projection 1's operand (blockIdx.y == 1; o0 again when n_proj == 1)
/// @param xv        the activation: bf16 or E4M3 [rows][K]
/// @param xsf       E4M3: the ue8m0 plane (pulsar_mx_sfoff); bf16: unread
/// @param kbp       E4M3: the plane's blocks per row
/// @param ids_src   source row per assignment
/// @param bounds    expert_bounds [E + 1]
/// @param E         experts
/// @param n_assign  assignments
/// @param K         input width, a multiple of 128
/// @param plan      the decode tile list to write, or none
template <int FMT, bool ROT>
__global__ void __launch_bounds__(256)
exl3_moe_prep_kernel(const prep_out o0, const prep_out o1, const void *__restrict__ xv, const uint8_t *__restrict__ xsf,
                     int kbp, const int32_t *__restrict__ ids_src, const int32_t *__restrict__ bounds, int E,
                     int n_assign, int K, const dec_plan plan) {
    if (plan.tiles && blockIdx.x == gridDim.x - 1) {
        if (blockIdx.y == 0) plan_decode_tiles(bounds, E, plan);
        return;
    }
    const prep_out &o = blockIdx.y ? o1 : o0;
    const void *__restrict__ table = o.table;
    __half *__restrict__ phi = o.phi;
    float *__restrict__ pinv = o.pinv;
    const int64_t task = (int64_t)blockIdx.x * 8 + (threadIdx.x >> 5);
    const int nblk = K >> 7;
    if (task >= (int64_t)n_assign * nblk) return;
    const int a = (int)(task / nblk), blk = (int)(task % nblk), lane = threadIdx.x & 31;
    if (a < bounds[0] || a >= bounds[E]) return;    /* outside the schedule (a pick < 0 sorts below expert 0) */
    const int k = blk * 128 + lane * 4;
    const int row = ids_src[a];
    float v[4];
    if constexpr (FMT == EXL3_MOE_ACT_BF16_ROWS) {
        const __nv_bfloat16 *x = static_cast<const __nv_bfloat16 *>(xv);
        const uint2 xw = *reinterpret_cast<const uint2 *>(x + (size_t)row * (size_t)K + (size_t)k);
        const float2 b01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&xw.x));
        const float2 b23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&xw.y));
        v[0] = b01.x; v[1] = b01.y; v[2] = b23.x; v[3] = b23.y;
    } else {
        /* the slot: [row][K] E4M3 and one ue8m0 byte per (row, 32-group); x 2^(byte - 127) by ldexpf,
         * exact (byte 0, an all-zero group's, included) */
        const uchar4 q = *reinterpret_cast<const uchar4 *>(static_cast<const uint8_t *>(xv) + (size_t)row * (size_t)K +
                                                           (size_t)k);
        const int e = (int)xsf[pulsar_mx_sfoff(row, k >> 5, kbp)] - 127;
        v[0] = ldexpf(exl3dev::e4m3_to_f32(q.x), e);
        v[1] = ldexpf(exl3dev::e4m3_to_f32(q.y), e);
        v[2] = ldexpf(exl3dev::e4m3_to_f32(q.z), e);
        v[3] = ldexpf(exl3dev::e4m3_to_f32(q.w), e);
    }
    if constexpr (ROT) {
        int lo = 0, hi = E - 1;                     /* the assignment's expert: bounds is E+1 ascending */
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (bounds[mid] <= a) lo = mid; else hi = mid - 1;
        }
        const void *const *pt = reinterpret_cast<const void *const *>(table);
        const __half *suh = reinterpret_cast<const __half *>(pt[2 * (size_t)lo + 1]);
        const uint2 sw = *reinterpret_cast<const uint2 *>(suh + k);
        const __half2 s01 = *reinterpret_cast<const __half2 *>(&sw.x), s23 = *reinterpret_cast<const __half2 *>(&sw.y);
        v[0] *= __low2float(s01); v[1] *= __high2float(s01); v[2] *= __low2float(s23); v[3] *= __high2float(s23);
        exl3dev::had128(v);
    }
    float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    int p = amax > 0.0f ? kHiExp - (((__float_as_int(amax) >> 23) & 0xff) - 126) : 0;
    p = max(-100, min(100, p));
    const float up = pow2f(p);
    uint32_t hw[2];
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const __half2 h2 = __floats2half2_rn(v[2 * q] * up, v[2 * q + 1] * up);
        hw[q] = *reinterpret_cast<const uint32_t *>(&h2);
    }
    const size_t off = (size_t)a * K + k;
    *reinterpret_cast<uint2 *>(phi + off) = make_uint2(hw[0], hw[1]);
    if (lane == 0) pinv[(size_t)a * nblk + blk] = pow2f(-p);
}

/* GEMM: CTA = (128 outputs, one work item); warp w = n16 tile w (outputs n0 + 16 w ..), all of the item's
 * LIVE m16 tiles.  Each warp decodes its own 8 k-tiles of the block into registers (no shared B: every
 * trellis tile is still decoded once per CTA) and applies each B fragment to every live m16 tile, whose
 * A comes from the staged planes by ldmatrix.  An item of mr rows runs ceil(mr / 16) m16 tiles, not 4:
 * at 4096 rows an expert averages ~80 assignments, so fixed 64-row tiles ran ~1.6x the MMAs.  The
 * per-(row, column) arithmetic is the 4 x 2 warp layout's exactly (a fresh fragment per k-tile,
 * scaled into the accumulator, blocks and k-tiles in order). */
/// The prefill grouped GEMM at rate K2.  Grid (M / 128, the work list's capacity), 256 threads, sizeof(Smem)
/// dynamic smem; CTAs past *n_work exit.
/// @param table    the expert table
/// @param phi      the PREP's fp16 plane [n_assign][K]
/// @param pinv     its inverse prescales [n_assign][K / 128]
/// @param ids_dst  output row per assignment
/// @param bounds   expert_bounds
/// @param work     the plan's items (expert, first assignment)
/// @param n_work   the item count
/// @param out      f32 [rows][M]: each assignment's unrotated z at row ids_dst
/// @param M        output width, a multiple of 128
/// @param K        input width, a multiple of 128
template <int K2>
__global__ void __launch_bounds__(kThreads, 2)
exl3_moe_gemm_kernel(const void *__restrict__ table, const __half *__restrict__ phi,
                     const float *__restrict__ pinv,
                     const int32_t *__restrict__ ids_dst, const int32_t *__restrict__ bounds,
                     const int2 *__restrict__ work, const int *__restrict__ n_work, float *__restrict__ out,
                     int M, int K) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem &S = *reinterpret_cast<Smem *>(smem_raw);
    if ((int)blockIdx.y >= *n_work) return;
    const int2 wk = work[blockIdx.y];
    const int e = wk.x, a0 = wk.y;
    const int mr = min(kBM, bounds[e + 1] - a0);
    const int mt_n = (mr + 15) >> 4;                   /* live m16 tiles */
    const int n0 = blockIdx.x * kBN, nblk = K >> 7;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    constexpr int W32 = exl3dev::Rate<K2>::words32;
    const uint32_t *tr = reinterpret_cast<const uint32_t *>(reinterpret_cast<const void *const *>(table)[2 * (size_t)e]);
    const int ntn = M >> 4, nt = (n0 >> 4) + warp;
    int lo[2], hi[2], sh[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) exl3dev::run4_window<K2>(8 * lane + 4 * h, lo[h], hi[h], sh[h]);
    const int g = lane >> 2, c2 = (lane & 3) * 2;
    const int arow = lane & 15, ahalf = lane >> 4;
    float acc[4][2][4];
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[m][h][i] = 0.0f;

    /* The pipeline: block b + 1's A goes to shared memory by cp.async and its raw trellis words to
     * registers while block b's MMAs run; a k-tile's words are decoded just before its MMAs.  Every
     * accumulator still takes its k-tiles 0..7 of each block in order, so the arithmetic is the
     * unpipelined kernel's (ncu: 66% of cycles had no eligible warp -- the loads were serialised
     * with the MMAs). */
    const int arows = mt_n * 16;
    auto issue_a = [&](int blk, int buf) {
        for (int c = tid; c < arows * 16; c += kThreads) {
            const int r = c >> 4, ch = c & 15;
            const bool real = r < mr;
            const __half *src = phi + (size_t)(a0 + (real ? r : 0)) * K + (size_t)blk * 128 + ch * 8;
            cp_async16(&S.a_hi[buf][r][ch * 8], src, real ? 16 : 0);
        }
#if PULSAR_SM80
        asm volatile("cp.async.commit_group;\n" ::: "memory");
#else
        __trap();
#endif
        if (tid < kBM) S.inv[buf][tid] = tid < mr ? pinv[(size_t)(a0 + tid) * nblk + blk] : 0.0f;
    };
    auto load_words = [&](int blk, uint32_t (&wd)[8][4]) {
#pragma unroll
        for (int kt = 0; kt < 8; ++kt) {
            const uint32_t *t = tr + ((size_t)(blk * 8 + kt) * (size_t)ntn + (size_t)nt) * W32;
            wd[kt][0] = t[lo[0]]; wd[kt][1] = t[hi[0]];
            wd[kt][2] = t[lo[1]]; wd[kt][3] = t[hi[1]];
        }
    };
    uint32_t wcur[8][4], wnxt[8][4];
    issue_a(0, 0);
    load_words(0, wcur);
    for (int blk = 0; blk < nblk; ++blk) {
        const int buf = blk & 1;
        const bool more = blk + 1 < nblk;
        if (more) {
            issue_a(blk + 1, buf ^ 1);
            load_words(blk + 1, wnxt);
#if PULSAR_SM80
            asm volatile("cp.async.wait_group 1;\n" ::: "memory");
#else
            __trap();
#endif
        } else {
#if PULSAR_SM80
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#else
            __trap();
#endif
        }
        __syncthreads();
        float ia[4], ib[4];
#pragma unroll
        for (int m = 0; m < 4; ++m) { ia[m] = S.inv[buf][m * 16 + g]; ib[m] = S.inv[buf][m * 16 + g + 8]; }
#pragma unroll
        for (int kt = 0; kt < 8; ++kt) {
            uint2 bw[2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                uint32_t st[4];
                exl3dev::run4_split<K2>(EXL3_RUN4_JOIN(wcur[kt][2 * h], wcur[kt][2 * h + 1], sh[h]), 8 * lane + 4 * h, st);
                bw[h] = make_uint2(exl3dev::mul1_pair(st[0], st[1]), exl3dev::mul1_pair(st[2], st[3]));
            }
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                if (m < mt_n) {                        /* warp-uniform */
                    uint32_t ahi[4];
                    ldmatrix_x4(ahi, &S.a_hi[buf][m * 16 + arow][kt * 16 + ahalf * 8]);
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        mma_16816(d, ahi, bw[h].x, bw[h].y);
                        float *ac = acc[m][h];
                        ac[0] = fmaf(d[0], ia[m], ac[0]);
                        ac[1] = fmaf(d[1], ia[m], ac[1]);
                        ac[2] = fmaf(d[2], ib[m], ac[2]);
                        ac[3] = fmaf(d[3], ib[m], ac[3]);
                    }
                }
            }
        }
        __syncthreads();                               /* buf is refilled two blocks on */
        if (more) {
#pragma unroll
            for (int kt = 0; kt < 8; ++kt)
#pragma unroll
                for (int i = 0; i < 4; ++i) wcur[kt][i] = wnxt[kt][i];
        }
    }
    /* acc[m][h][i]: row 16 m + g + 8 (i >> 1), column n0 + 16 warp + 8 h + c2 + (i & 1) */
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = m * 16 + g + 8 * (i >> 1);
                if (r < mr) {
                    const int col = n0 + 16 * warp + 8 * h + c2 + (i & 1);
                    out[(size_t)ids_dst[a0 + r] * (size_t)M + (size_t)col] = acc[m][h][i];
                }
            }
}

/* ---------------------------------------------------------------------- */
/* the DECODE GEMM (task #46)                                              */

constexpr int kDecWarps = 4;                  ///< the CTA's k-range split four ways, summed in warp order
constexpr int kDecNT    = 4;                  ///< n16 tiles per warp: every warp covers the CTA's 64 outputs
/// Outputs per decode CTA (64).
constexpr int kDecBN    = kDecNT * 16;
constexpr int kDecMinWarpTiles = 16;          ///< k-tiles a warp runs at least: the split count's floor
constexpr int kDecMinCtas = 4;                ///< CTAs per SM the register budget is held to (128, no spill)
/* Measured (task #46, V4 / Qwen shapes, 1..40 rows): a floor of 8 k-tiles 3-5% slower than 16 or 32.
 * L297 (tests/exl3_moe_decode_bench, DRAM-cold, Nemotron / Qwen / V4 / V4.1 shapes at 1..16 rows; every
 * variant's outputs byte-identical): with the planned tiles, 4 n16 tiles per warp at 4 CTAs per SM (128
 * registers, no spill) beat 8 at 3 (168 registers, 24-64 bytes spilled at every rate) by 0-3% on Nemotron,
 * 2-5% on Qwen and 1-6% on DeepSeek; 2 n16 tiles at 5-6 CTAs, or two k-tiles in flight, won up to 4% on
 * Nemotron / Qwen and lost 5-20% on DeepSeek.  The routed reads run at 220-230 GB/s from 4 rows (GB10's
 * streaming read: ~236). */

/* The K splits of a K-wide projection: as many as keep every warp at >= kDecMinWarpTiles k-tiles, dividing
 * the k-tiles evenly.  A function of K ALONE -- never of the row count -- so a row's bits do not depend on
 * how many rows ride with it. */
/// @param K  input width, a multiple of 128
/// @return   the split count SK, >= 1
int decode_splits(int K) {
    const int q = (K >> 4) / kDecWarps;       /* k-tiles per warp unsplit; K % 128 makes it whole */
    int best = 1;
    for (int s = 1; s <= q; ++s)
        if (q % s == 0 && q / s >= kDecMinWarpTiles) best = s;
    return best;
}

/* One projection of a decode launch: its expert table, the PREP's operand for it, its split partials
 * z [splits][rows][M] (rows = the schedule's ids_dst). */
/// One projection of a decode launch.
struct dec_mat {
    /// The expert table.
    const void *table;
    /// The PREP's fp16 plane [n_assign][K].
    const __half *phi;
    /// Its inverse prescales [n_assign][K / 128].
    const float *pinv;
    /// The split partials f32 [SK][rows][M].
    float *z;
};

/* A tile's words as the lanes hold them.  Where one 64-bit window carries a lane's eight states (run8: K <= 4)
 * lane l holds word l, one coalesced load per tile, and takes its window's two words from its neighbours by
 * shuffle (TensorFold's experts_grouped.cuh load).  Where it takes two windows (K5, K6: 40 / 48-word tiles, two
 * words a lane) the four shuffles of each tile_word pair cost more than they save: the lane loads its four
 * window words itself (L1 hits on the tile its warp reads) -- measured on Qwen's K5 down, 640 -> 2560. */
/// The decode kernel's per-rate tile geometry (see the comment above).
template <int K2> struct DecGeo {
    /// uint32 words per tile.
    static constexpr int W32 = exl3dev::Rate<K2>::words32;
    /// Words per lane when a warp loads a tile coalesced, ceil(W32 / 32).
    static constexpr int LW = (W32 + 31) / 32;
    static constexpr bool R8 = exl3dev::run8_slack(K2) >= 0;   ///< one window per lane per tile, else two
    static constexpr int SW = R8 ? 1 : 4;                       ///< words a lane holds per tile
    static_assert(!R8 || LW == 1, "a run8 rate's tile is one word per lane");
};
/* word idx of a tile whose lane l holds word l (a run8 rate's tile is at most 32 words: DecGeo static_assert) */
/// @param w    this lane's word of the tile
/// @param idx  the word wanted, 0..31
/// @return     word idx of the tile, from lane idx
__device__ __forceinline__ uint32_t tile_word(uint32_t w, int idx) { return __shfl_sync(0xffffffffu, w, idx); }

/* CTA = (64 outputs, tile blockIdx.y of the plan -- an m16 tile of one expert's assignments --, one K split of
 * one projection).  The grid spans an upper bound (every assignment); the CTAs past the plan's tiles have nothing
 * to do and leave at once.  Warp w runs k-tiles [kt0, kt0 + per_warp) of the split, every k-tile: the A
 * fragment straight from the PREP's plane (rows past the tile's end are zero), each of its kDecNT n16 trellis
 * tiles decoded into mma B registers (mul1_pair: the prefill arm's decode bit for bit), one MMA per n8 half
 * into a fresh fragment scaled by the row's inverse prescale into the accumulator -- the prefill arm's
 * per-k-tile arithmetic.  The next k-tile's words, A and prescales load while this one computes.  The four
 * warps' partials are summed in warp order and written to the split's plane; the consumer (fold / sum) adds
 * the splits in split order. */
/// The decode split-K GEMM at rate K2.  Grid (M / 64, n_assign, SK x n_proj), 128 threads, 8 KB static smem (the warp reduction).
/// @param m0       projection 0
/// @param m1       projection 1 (blockIdx.z / SK == 1)
/// @param ids_dst  output row per assignment
/// @param bounds   expert_bounds
/// @param tiles    the plan's tiles (expert, first assignment)
/// @param n_tiles  the tile count
/// @param M        output width, a multiple of 64
/// @param K        input width, a multiple of 128
/// @param SK       splits (decode_splits(K))
/// @param z_split  floats between consecutive splits' planes
template <int K2>
__global__ void __launch_bounds__(kDecWarps * 32, kDecMinCtas)
exl3_moe_decode_kernel(const dec_mat m0, const dec_mat m1, const int32_t *__restrict__ ids_dst,
                       const int32_t *__restrict__ bounds, const int2 *__restrict__ tiles,
                       const int *__restrict__ n_tiles, int M, int K, int SK, int64_t z_split) {
    using G = DecGeo<K2>;
    constexpr int NT = kDecNT, BN = kDecBN;
    __shared__ float red[kDecWarps][kDecRows / 2][BN];
    if ((int)blockIdx.y >= *n_tiles) return;
    const int2 tile = tiles[blockIdx.y];
    const int lo = tile.x, a0 = tile.y;               /* the tile's expert and its first assignment */
    const int mr = min(kDecRows, bounds[lo + 1] - a0);
    const int split = (int)blockIdx.z % SK, mat = (int)blockIdx.z / SK;
    const dec_mat &p = mat ? m1 : m0;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, c2 = (lane & 3) * 2;
    const int KT = K >> 4, per_warp = KT / SK / kDecWarps, nblk = K >> 7;
    const int kt0 = split * (KT / SK) + warp * per_warp;
    const int ntn = M >> 4, nt0 = (int)blockIdx.x * NT;
    const uint32_t *tr = reinterpret_cast<const uint32_t *>(reinterpret_cast<const void *const *>(p.table)[2 * (size_t)lo]) +
                         ((size_t)kt0 * (size_t)ntn + (size_t)nt0) * G::W32 + lane;
    const size_t kstride = (size_t)ntn * G::W32;
    /* the lane's windows: positions 8 lane .. 8 lane + 7 (n8 half 0 = the first four, half 1 the last four) */
    constexpr int NWIN = G::R8 ? 1 : 2;
    int wlo[NWIN], whi[NWIN], wsh[NWIN];
    if constexpr (G::R8) {
        exl3dev::run8_window<K2>(8 * lane, wlo[0], whi[0], wsh[0]);
    } else {
#pragma unroll
        for (int h = 0; h < 2; ++h) exl3dev::run4_window<K2>(8 * lane + 4 * h, wlo[h], whi[h], wsh[h]);
    }
    /* the lane's two A rows: g and g + 8 of the tile */
    const bool oka = g < mr, okb = g + 8 < mr;
    const size_t ra = (size_t)a0 + (oka ? g : 0), rb = (size_t)a0 + (okb ? g + 8 : 0);
    const __half *xa = p.phi + ra * K + c2, *xb = p.phi + rb * K + c2;
    const float *ia_p = p.pinv + ra * nblk, *ib_p = p.pinv + rb * nblk;

    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][h][j] = 0.0f;

    struct stage {
        uint32_t w[NT][G::SW];
        uint32_t a[4];
        float ia, ib;
    };
    auto load = [&](int it, stage &s) {
        const uint32_t *t = tr + (size_t)it * kstride;
        if constexpr (G::R8) {
#pragma unroll
            for (int i = 0; i < NT; ++i) s.w[i][0] = (G::W32 == 32 || lane < G::W32) ? __ldg(t + i * G::W32) : 0u;
        } else {
            t -= lane;                                   /* the tile's own words */
#pragma unroll
            for (int i = 0; i < NT; ++i) {
                s.w[i][0] = __ldg(t + i * G::W32 + wlo[0]); s.w[i][1] = __ldg(t + i * G::W32 + whi[0]);
                s.w[i][2] = __ldg(t + i * G::W32 + wlo[1]); s.w[i][3] = __ldg(t + i * G::W32 + whi[1]);
            }
        }
        const int k = (kt0 + it) * 16, b = (kt0 + it) >> 3;
        s.a[0] = oka ? *reinterpret_cast<const uint32_t *>(xa + k) : 0u;
        s.a[1] = okb ? *reinterpret_cast<const uint32_t *>(xb + k) : 0u;
        s.a[2] = oka ? *reinterpret_cast<const uint32_t *>(xa + k + 8) : 0u;
        s.a[3] = okb ? *reinterpret_cast<const uint32_t *>(xb + k + 8) : 0u;
        s.ia = oka ? ia_p[b] : 0.0f;
        s.ib = okb ? ib_p[b] : 0.0f;
    };
    stage c, nxt;
    load(0, c);
    for (int it = 0; it < per_warp; ++it) {
        if (it + 1 < per_warp) load(it + 1, nxt);
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            uint32_t b[2][2];
            if constexpr (G::R8) {
                uint32_t st[8];
                exl3dev::run8_split<K2>(EXL3_RUN4_JOIN(tile_word(c.w[i][0], wlo[0]), tile_word(c.w[i][0], whi[0]),
                                                       wsh[0]), 8 * lane, st);
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    b[h][0] = exl3dev::mul1_pair(st[4 * h], st[4 * h + 1]);
                    b[h][1] = exl3dev::mul1_pair(st[4 * h + 2], st[4 * h + 3]);
                }
            } else {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    uint32_t st[4];
                    exl3dev::run4_split<K2>(EXL3_RUN4_JOIN(c.w[i][2 * h], c.w[i][2 * h + 1], wsh[h]), 8 * lane + 4 * h, st);
                    b[h][0] = exl3dev::mul1_pair(st[0], st[1]);
                    b[h][1] = exl3dev::mul1_pair(st[2], st[3]);
                }
            }
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                mma_16816(d, c.a, b[h][0], b[h][1]);
                float *ac = acc[i][h];
                ac[0] = fmaf(d[0], c.ia, ac[0]);
                ac[1] = fmaf(d[1], c.ia, ac[1]);
                ac[2] = fmaf(d[2], c.ib, ac[2]);
                ac[3] = fmaf(d[3], c.ib, ac[3]);
            }
        }
        c = nxt;
    }
    /* acc[i][h][j]: row g + 8 (j >> 1), column 16 i + 8 h + c2 + (j & 1) of the CTA's BN.  The warps' partials
     * are summed in warp order, the tile's rows 0..7 then 8..15 (the second pass only when the tile has them):
     * half the shared memory, so it does not cap the CTAs an SM holds. */
    float *z = p.z + (size_t)split * (size_t)z_split;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        if (half * 8 >= mr) break;                   /* CTA-uniform */
        if (half) __syncthreads();                   /* the first pass's readers are done */
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int j = 0; j < 2; ++j) red[warp][g][16 * i + 8 * h + c2 + j] = acc[i][h][2 * half + j];
        __syncthreads();
        const int rows = min(8, mr - half * 8);
        for (int idx = threadIdx.x; idx < rows * BN; idx += kDecWarps * 32) {
            const int r = idx / BN, col = idx - r * BN;
            float sum = red[0][r][col];
#pragma unroll
            for (int w = 1; w < kDecWarps; ++w) sum += red[w][r][col];
            z[(size_t)ids_dst[a0 + half * 8 + r] * (size_t)M + (size_t)(nt0 * 16 + col)] = sum;
        }
    }
}

/// Launch exl3_moe_decode_kernel<K2>; the arguments are the kernel's.
/// @param grid     (M / 64, n_assign, SK x n_proj)
/// @param stream   the launch stream
/// @param m0       projection 0
/// @param m1       projection 1
/// @param ids_dst  output row per assignment
/// @param bounds   expert_bounds
/// @param plan     the tile list the PREP wrote
/// @param M        output width
/// @param K        input width
/// @param SK       splits
/// @param z_split  floats between splits
/// @return         true
template <int K2>
bool launch_decode(const dim3 &grid, cudaStream_t stream, const dec_mat &m0, const dec_mat &m1, const int32_t *ids_dst,
                   const int32_t *bounds, const dec_plan &plan, int M, int K, int SK, int64_t z_split) {
    exl3_moe_decode_kernel<K2><<<grid, kDecWarps * 32, 0, stream>>>(m0, m1, ids_dst, bounds, plan.tiles, plan.n_tiles,
                                                                    M, K, SK, z_split);
    return true;
}

/// Launch exl3_moe_gemm_kernel<K2>, opting it into sizeof(Smem) dynamic smem on first use.
/// @param grid     (M / 128, the work list's capacity)
/// @param stream   the launch stream
/// @param table    the expert table
/// @param phi      the PREP's plane
/// @param pinv     its inverse prescales
/// @param ids_dst  output row per assignment
/// @param bounds   expert_bounds
/// @param work     the work list
/// @param n_work   its count
/// @param out      f32 [rows][M]
/// @param M        output width
/// @param K        input width
/// @return         false when the smem attribute is refused
template <int K2>
bool launch_gemm(const dim3 &grid, cudaStream_t stream, const void *table, const __half *phi,
                 const float *pinv, const int32_t *ids_dst, const int32_t *bounds, const int2 *work,
                 const int *n_work, float *out, int M, int K) {
    static bool attr = false;
    if (!attr) {
        if (cudaFuncSetAttribute(exl3_moe_gemm_kernel<K2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)sizeof(Smem)) != cudaSuccess) return false;
        attr = true;
    }
    exl3_moe_gemm_kernel<K2><<<grid, kThreads, sizeof(Smem), stream>>>(table, phi, pinv, ids_dst, bounds, work,
                                                                       n_work, out, M, K);
    return true;
}

/* The PREP of n_proj (1 or 2) projections over n_assign assignments, in the activation's own format; with a plan
 * (plan.tiles: a decode launch) one more CTA writes the decode GEMM's tile list. */
/// @param rotate    the projection rotates its input (x * suh, H128)
/// @param stream    the launch stream
/// @param o0        projection 0's operand
/// @param o1        projection 1's operand (o0 again when n_proj == 1)
/// @param n_proj    1 or 2
/// @param act       the activation as its producer emitted it
/// @param ids_src   source row per assignment
/// @param bounds    expert_bounds
/// @param E         experts
/// @param n_assign  assignments
/// @param K         input width
/// @param plan      the decode tile list to write, or none
void launch_prep(bool rotate, cudaStream_t stream, const prep_out &o0, const prep_out &o1, int n_proj,
                 const exl3_moe_act &act, const int32_t *ids_src, const int32_t *bounds, int E, int n_assign, int K,
                 const dec_plan &plan) {
    const uint8_t *sf = static_cast<const uint8_t *>(act.sf);
    const int64_t tasks = (int64_t)n_assign * (K >> 7);
    const dim3 grid((unsigned)((tasks + 7) / 8 + (plan.tiles ? 1 : 0)), (unsigned)n_proj);
    const bool e4m3 = act.format == EXL3_MOE_ACT_E4M3_SLOT;
    if (e4m3 && rotate)
        exl3_moe_prep_kernel<EXL3_MOE_ACT_E4M3_SLOT, true><<<grid, 256, 0, stream>>>(o0, o1, act.x, sf, act.kbp, ids_src,
                                                                                    bounds, E, n_assign, K, plan);
    else if (e4m3)
        exl3_moe_prep_kernel<EXL3_MOE_ACT_E4M3_SLOT, false><<<grid, 256, 0, stream>>>(o0, o1, act.x, sf, act.kbp,
                                                                                     ids_src, bounds, E, n_assign, K, plan);
    else if (rotate)
        exl3_moe_prep_kernel<EXL3_MOE_ACT_BF16_ROWS, true><<<grid, 256, 0, stream>>>(o0, o1, act.x, sf, act.kbp, ids_src,
                                                                                    bounds, E, n_assign, K, plan);
    else
        exl3_moe_prep_kernel<EXL3_MOE_ACT_BF16_ROWS, false><<<grid, 256, 0, stream>>>(o0, o1, act.x, sf, act.kbp,
                                                                                     ids_src, bounds, E, n_assign, K, plan);
}

/// @param b  bytes
/// @return   b rounded up to a multiple of 256
size_t up256(size_t b) { return (b + 255u) & ~(size_t)255u; }

/* The workspace's three slices, in order: the fp16 plane, the inverse prescales, the work list + its count. */
/// The prefill workspace's byte offsets.
struct ws_layout {
    /** @var inv
     *  offset of the inverse prescales f32 [n_assign][K / 128] */
    /** @var work
     *  offset of the work list int2 [cap], then its int count */
    /** @var total
     *  the workspace's bytes */
    /// Offset of the fp16 plane [n_assign][K] (0).
    size_t plane, inv, work, total;
    /// The work list's capacity: n_assign / 64 + n_experts + 1.
    int64_t cap;
};
/// @param n_assign   assignments
/// @param K          input width
/// @param n_experts  experts
/// @return           the layout, each slice 256-aligned
ws_layout layout_for(int64_t n_assign, int K, int n_experts) {
    ws_layout l;
    l.cap = n_assign / kBM + n_experts + 1;
    l.plane = up256((size_t)n_assign * (size_t)K * sizeof(__half));
    l.inv = up256((size_t)n_assign * (size_t)(K >> 7) * sizeof(float));
    l.work = up256((size_t)l.cap * sizeof(int2) + sizeof(int));
    l.total = l.plane + l.inv + l.work;
    return l;
}

} // namespace

size_t exl3_moe_prefill_ws_bytes(int64_t n_assign, int K, int n_experts) {
    if (n_assign <= 0 || K <= 0 || n_experts <= 0) return 0;
    return layout_for(n_assign, K, n_experts).total;
}

int exl3_moe_prefill_launch(const void *table, int k2, bool rotate_input, const exl3_moe_act &act,
                            const int32_t *ids_dst, const int32_t *ids_src, const int32_t *expert_bounds,
                            float *out, int M, int K, int64_t n_assign, int n_experts,
                            void *ws, size_t ws_bytes, cudaStream_t stream) {
    PULSAR_REQUIRE_CC(80, "the EXL3 MoE prefill GEMM (m16n8k16, cp.async)", -1);   /* L321: refuse below the lane's arch, before its kernels trap */
    const char *tag = "exl3_moe_prefill_launch";
    const bool e4m3 = act.format == EXL3_MOE_ACT_E4M3_SLOT;
    if (!table || !act.x || !ids_dst || !ids_src || !expert_bounds || !out || !ws || n_assign <= 0 ||
        n_experts <= 0 || n_experts > kMaxE || M <= 0 || M % kBN || K <= 0 || K % EXL3_HAD_BLOCK ||
        (act.format != EXL3_MOE_ACT_BF16_ROWS && !e4m3) || (e4m3 && (!act.sf || act.kbp < K / 32)) ||
        (!e4m3 && act.sf)) {
        fprintf(stderr, "%s: bad arguments (M=%d K=%d n_assign=%lld E=%d act format %d sf=%p kbp=%d; M %% %d, "
                        "K %% 128, E <= %d) -- refusing\n", tag, M, K, (long long)n_assign, n_experts, act.format,
                act.sf, act.kbp, kBN, kMaxE);
        return -1;
    }
    if (!exl3_arm_has_rate(EXL3_ARM_MOE_PREFILL, k2)) {
        fprintf(stderr, "%s: rate k2=%d has no prefill instance -- refusing\n", tag, k2);
        return -1;
    }
    const ws_layout l = layout_for(n_assign, K, n_experts);
    if (l.cap > kPlanCap) {
        fprintf(stderr, "%s: %lld assignments over %d experts need %lld work items > %d -- refusing\n",
                tag, (long long)n_assign, n_experts, (long long)l.cap, kPlanCap);
        return -1;
    }
    if (ws_bytes < l.total || ((uintptr_t)ws & 255u)) {
        fprintf(stderr, "%s: workspace %zu B at %p < %zu B (256-aligned) -- refusing\n", tag, ws_bytes, ws, l.total);
        return -1;
    }
    static bool announced[2] = {false, false};
    if (!announced[e4m3]) {
        announced[e4m3] = true;
        fprintf(stderr, "pulsar: EXL3 MoE prefill arm = fp16 tensor-core GEMM over %s rows (first call: K=%g, "
                        "%d x %d, %lld assignments over %d experts)\n", e4m3 ? "E4M3-slot" : "bf16",
                k2 / 2.0, K, M, (long long)n_assign, n_experts);
    }
    uint8_t *base = static_cast<uint8_t *>(ws);
    __half *phi = reinterpret_cast<__half *>(base);
    float *pinv = reinterpret_cast<float *>(base + l.plane);
    int2 *work = reinterpret_cast<int2 *>(base + l.plane + l.inv);
    int *n_work = reinterpret_cast<int *>(work + l.cap);
    int threads = 32;
    while (threads < n_experts) threads <<= 1;
    exl3_moe_plan_kernel<<<1, threads, 0, stream>>>(expert_bounds, n_experts, work, n_work, (int)l.cap);
    const prep_out po{table, phi, pinv};
    launch_prep(rotate_input, stream, po, po, 1, act, ids_src, expert_bounds, n_experts, (int)n_assign, K, dec_plan{});
    const dim3 grid((unsigned)(M / kBN), (unsigned)l.cap, 1);
    /* the rates exl3_arm_has_rate(EXL3_ARM_MOE_PREFILL) names: every rate a routed projection kind
     * admits -- k2 = 4 / 5 / 6 DeepSeek's (K2, 2.5, 3) and Qwen's
     * MTP experts; 8 / 10 / 12 Qwen's trunk (4.05 / 3.05 / 6.05 bpw) */
    bool ok = false;
    switch (k2) {
    case 4:  ok = launch_gemm<4>(grid, stream, table, phi, pinv, ids_dst, expert_bounds, work, n_work, out, M, K); break;
    case 5:  ok = launch_gemm<5>(grid, stream, table, phi, pinv, ids_dst, expert_bounds, work, n_work, out, M, K); break;
    case 6:  ok = launch_gemm<6>(grid, stream, table, phi, pinv, ids_dst, expert_bounds, work, n_work, out, M, K); break;
    case 8:  ok = launch_gemm<8>(grid, stream, table, phi, pinv, ids_dst, expert_bounds, work, n_work, out, M, K); break;
    case 10: ok = launch_gemm<10>(grid, stream, table, phi, pinv, ids_dst, expert_bounds, work, n_work, out, M, K); break;
    case 12: ok = launch_gemm<12>(grid, stream, table, phi, pinv, ids_dst, expert_bounds, work, n_work, out, M, K); break;
    default: break;
    }
    const cudaError_t err = cudaGetLastError();
    if (!ok || err != cudaSuccess) {
        fprintf(stderr, "%s: launch failed: %s\n", tag, ok ? cudaGetErrorString(err) : "shared-memory attribute / rate");
        return -3;
    }
    return 0;
}

namespace {

/* One decode projection's workspace: the PREP's fp16 plane and inverse prescales, then the split partials. */
/// One decode projection's workspace byte offsets.
struct dec_layout {
    /** @var inv
     *  offset of the inverse prescales f32 [n_assign][K / 128] */
    /** @var part
     *  offset of the split partials f32 [SK][n_assign][M] */
    /** @var one
     *  the projection's bytes */
    /// Offset of the fp16 plane [n_assign][K] (0).
    size_t plane, inv, part, one;
};
/// @param n_assign  assignments
/// @param K         input width
/// @param M         output width
/// @return          the layout, each slice 256-aligned
dec_layout dec_layout_for(int64_t n_assign, int K, int M) {
    dec_layout l;
    l.plane = up256((size_t)n_assign * (size_t)K * sizeof(__half));
    l.inv = up256((size_t)n_assign * (size_t)(K >> 7) * sizeof(float));
    l.part = up256((size_t)decode_splits(K) * (size_t)n_assign * (size_t)M * sizeof(float));
    l.one = l.plane + l.inv + l.part;
    return l;
}

} // namespace

/* The decode tile list (dec_plan) after the projections' slices: n_assign tiles at most, then the count. */
/// @param n_assign assignments.
/// @return bytes, 256-padded.
size_t dec_plan_bytes(int64_t n_assign) { return up256((size_t)n_assign * sizeof(int2) + sizeof(int)); }

size_t exl3_moe_decode_ws_bytes(int n_proj, int64_t n_assign, int K, int M) {
    if (n_proj < 1 || n_proj > 2 || n_assign <= 0 || K <= 0 || M <= 0 || K % EXL3_HAD_BLOCK) return 0;
    return (size_t)n_proj * dec_layout_for(n_assign, K, M).one + dec_plan_bytes(n_assign);
}

int exl3_moe_decode_launch(const void *table0, const void *table1, int k2, bool rotate_input, const exl3_moe_act &act,
                           const int32_t *ids_dst, const int32_t *ids_src, const int32_t *expert_bounds, int M, int K,
                           int64_t n_assign, int n_experts, void *ws, size_t ws_bytes, exl3_moe_partials *out,
                           cudaStream_t stream) {
    PULSAR_REQUIRE_CC(80, "the EXL3 MoE decode (m16n8k16)", -1);   /* L321: refuse below the lane's arch, before its kernels trap */
    const char *tag = "exl3_moe_decode_launch";
    const bool e4m3 = act.format == EXL3_MOE_ACT_E4M3_SLOT;
    const int n_proj = table1 ? 2 : 1;
    if (!table0 || !act.x || !ids_dst || !ids_src || !expert_bounds || !ws || !out || n_assign <= 0 ||
        n_assign > EXL3_MOE_DECODE_MAX_ASSIGN || n_experts <= 0 || M <= 0 || M % EXL3_HAD_BLOCK || K <= 0 ||
        K % EXL3_HAD_BLOCK || (act.format != EXL3_MOE_ACT_BF16_ROWS && !e4m3) ||
        (e4m3 && (!act.sf || act.kbp < K / 32)) || (!e4m3 && act.sf) || (table1 && !rotate_input)) {
        fprintf(stderr, "%s: bad arguments (M=%d K=%d n_assign=%lld E=%d act format %d sf=%p kbp=%d projections %d; "
                        "M %% %d, K %% 128, n_assign <= %lld, a pair rotates) -- refusing\n", tag, M, K,
                (long long)n_assign, n_experts, act.format, act.sf, act.kbp, n_proj, EXL3_HAD_BLOCK,
                (long long)EXL3_MOE_DECODE_MAX_ASSIGN);
        return -1;
    }
    if (!exl3_arm_has_rate(EXL3_ARM_MOE_PREFILL, k2)) {
        fprintf(stderr, "%s: rate k2=%d has no instance -- refusing\n", tag, k2);
        return -1;
    }
    const dec_layout l = dec_layout_for(n_assign, K, M);
    const size_t need = (size_t)n_proj * l.one + dec_plan_bytes(n_assign);
    if (ws_bytes < need || ((uintptr_t)ws & 255u)) {
        fprintf(stderr, "%s: workspace %zu B at %p < %zu B (256-aligned) -- refusing\n", tag, ws_bytes, ws, need);
        return -1;
    }
    const int SK = decode_splits(K);
    static bool announced[2] = {false, false};
    if (!announced[e4m3]) {
        announced[e4m3] = true;
        fprintf(stderr, "pulsar: EXL3 MoE decode arm = fp16 tensor-core split-K GEMM over %s rows (first call: K=%g, "
                        "%d x %d, %d splits, %lld assignments over %d experts)\n", e4m3 ? "E4M3-slot" : "bf16",
                k2 / 2.0, K, M, SK, (long long)n_assign, n_experts);
    }
    uint8_t *base = static_cast<uint8_t *>(ws);
    prep_out po[2];
    dec_mat dm[2];
    for (int j = 0; j < 2; ++j) {
        uint8_t *b = base + (size_t)(j < n_proj ? j : 0) * l.one;
        po[j] = prep_out{j && table1 ? table1 : table0, reinterpret_cast<__half *>(b), reinterpret_cast<float *>(b + l.plane)};
        dm[j] = dec_mat{po[j].table, po[j].phi, po[j].pinv, reinterpret_cast<float *>(b + l.plane + l.inv)};
    }
    int2 *tiles = reinterpret_cast<int2 *>(base + (size_t)n_proj * l.one);
    const dec_plan plan{tiles, reinterpret_cast<int *>(tiles + n_assign)};
    launch_prep(rotate_input, stream, po[0], po[1], n_proj, act, ids_src, expert_bounds, n_experts, (int)n_assign, K,
                plan);
    const int64_t z_split = n_assign * (int64_t)M;
    const dim3 grid((unsigned)(M / kDecBN), (unsigned)n_assign, (unsigned)(SK * n_proj));
    bool ok = false;
    switch (k2) {
    case 4:  ok = launch_decode<4>(grid, stream, dm[0], dm[1], ids_dst, expert_bounds, plan, M, K, SK, z_split); break;
    case 5:  ok = launch_decode<5>(grid, stream, dm[0], dm[1], ids_dst, expert_bounds, plan, M, K, SK, z_split); break;
    case 6:  ok = launch_decode<6>(grid, stream, dm[0], dm[1], ids_dst, expert_bounds, plan, M, K, SK, z_split); break;
    case 8:  ok = launch_decode<8>(grid, stream, dm[0], dm[1], ids_dst, expert_bounds, plan, M, K, SK, z_split); break;
    case 10: ok = launch_decode<10>(grid, stream, dm[0], dm[1], ids_dst, expert_bounds, plan, M, K, SK, z_split); break;
    case 12: ok = launch_decode<12>(grid, stream, dm[0], dm[1], ids_dst, expert_bounds, plan, M, K, SK, z_split); break;
    default: break;
    }
    const cudaError_t err = cudaGetLastError();
    if (!ok || err != cudaSuccess) {
        fprintf(stderr, "%s: launch failed: %s\n", tag, ok ? cudaGetErrorString(err) : "rate");
        return -3;
    }
    *out = exl3_moe_partials{{dm[0].z, n_proj == 2 ? dm[1].z : nullptr}, SK, z_split};
    return 0;
}

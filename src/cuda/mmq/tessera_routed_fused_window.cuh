// VENDORED from Tessera (https://github.com/RobTand/tessera), src/tessera/serving/csrc/routed_fused_window.cu
// at upstream 1381c3b7 (2026-09-29), file sha256 5892fd1998b95168db17dd2405950fd09687f8eb291906868a16ba691698da6b.
// Made with Tessera by Robert Tand - https://github.com/RobTand/tessera
// License: MIT + Tessera Attribution Addendum 1.0 (LicenseRef-Tessera-Attribution-1.0); see src/cuda/mmq/VENDOR-TESSERA.md.
//
// pulsar edits (host glue only -- every device function is byte-for-byte upstream):
//   * the torch/c10 #includes (upstream lines 58-61) are dropped;
//   * the three c10 CUDA checks and the one TORCH_CHECK in max_dynamic_smem_bytes / launch_pair / launch call
//     the includer's hooks TESSERA_HOST_CUDA(expr) and TESSERA_HOST_REFUSE_PAIR(mode, r_lo, two, tile_words, K);
//   * the torch host entries and PYBIND module (upstream lines 1158-1177 and 1208-1520) are dropped; pulsar's
//     launcher (pulsar_tessera.cu) replaces them.  dense_reduce_kernel (upstream 1178-1206) is kept verbatim.
// The includer defines TESSERA_ROUTED_FUSED_FP8 and the two hooks before including this file, and compiles the
// translation unit WITHOUT --use_fast_math (upstream builds -O3 -lineinfo; the SwiGLU epilogue calls expf and
// divides, which fast-math would approximate).

// The fused routed window MoE kernel (tessera#640): one persistent,
// warp-specialised CUDA kernel serves a routed expert stack's gate/up
// projection WITH its SwiGLU epilogue, and a second launch of the same kernel
// serves the down projection with a deterministic route-sorted output.
//
// WHAT IT COMPUTES.  The same functions of the wire as the Triton grouped
// window GEMM (``tessera.window_gemm_grouped``): the 14-bit window state of
// row ``n`` in column ``k`` is the last 14 bits of that column's MSB-first bit
// stream (``rate`` bits per row, any rate 1..8, per column -- the run table's
// one rate or two ADJACENT rates, the pair bracketing the stack's root) ending
// after row ``n``, looked up in the expert's table.  The BF16 (value) family
// is FOLDED -- ``bf16(table[state] * row_scale[n])`` before the dot, no
// epilogue scale -- and the E4M3 family
// runs the epilogue arithmetic ``(acc * a_scale[row]) * row_scale[n]``, both
// in the exact fp32 operation order of the legacy kernel.  The E4M3 table
// entry ``native[codes[state]]`` is composed on the host into an f16 table
// (exact: e4m3 -> f16 is lossless) and the fp8 activation is converted to f16
// on the way into shared memory (also exact), so the E4M3 stack runs on the
// f16 tensor-core instruction with f32 accumulation.
//
// HOW IT IS SCHEDULED.  Work items are ``(expert, n-block, route superblock)``
// triples; their count is a function of the routing, computed on the device
// (``item_off`` = prefix sum of ceil(routes_e / 64)) and claimed through one
// device counter, so the grid is the SM count and no host synchronisation
// exists.  Eight producer warps stream the packed words with ``cp.async``,
// decode them through the shared-memory table into a swizzled 16-bit B tile
// and stage the A tile; eight consumer warps run ``ldmatrix`` + ``mma.sync``
// on a double-buffered stage behind named barriers, and write the epilogue.
// Every weight is decoded once per (item, superblock) and reused across the
// superblock's 64 routes.
//
// DETERMINISM.  The down projection writes each route's bf16-rounded,
// route-weighted row into a ``[routes, H]`` buffer indexed by the route's
// original position; ``token_sum`` then adds each token's ``top_k`` rows in
// fixed route order in fp32 and rounds once.  No atomics anywhere, so two
// runs are bitwise equal.  The device work counter is zeroed by the caller
// inside the same stream (a graph-captured memset), so a captured forward
// replays.
//
// THE DENSE CASE.  A dense Linear is the E = 1, top_k = 1, unweighted case of
// the down projection: ``routed_fused_kernel<FP8, 2, DENSE=true, SPLIT>`` reads
// the same words/table/scale planes for one "expert", takes row ``m`` of ``x``
// as route ``m`` (no routing tables are read), and writes ``y[m, n]`` straight
// into a column slice of the module's output (``out_stride`` is the slice's row
// stride), so a merged Linear's roles are one launch each and no concatenation
// follows.  When the item count ``ceil(M / 64) * N / 128`` would leave SMs idle
// (decode: M <= 64 on a 4096-row role is 32 items for 48 SMs) the K range is
// split ``S`` ways (``k_split``); each split accumulates its chunk range into
// an fp32 workspace ``[S, M, N]`` and ``dense_reduce_kernel`` sums the ``S``
// partials in fixed order and applies the epilogue -- deterministic, and the
// same fp32 operation order as the unsplit epilogue once the sum is formed.
// The dense identity is ``tessera::fused_window_dense`` (``serving.native_
// window``), decoders ``native_fused_window_dense`` / ``..._folded``.
//
// The Python owner is ``tessera.routed_fused``; the contract publishes this
// file as two ``native_extensions`` entries (one per family, see ``ext``).

// (upstream lines 58-61: the torch / c10 includes -- dropped, see the header above)
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>
#include <limits>

#ifndef TESSERA_ROUTED_FUSED_FP8
#error "compile with -DTESSERA_ROUTED_FUSED_FP8=0 (the value family's library) or =1 (the E4M3 family's)"
#endif

namespace {

// One library per family (``ext.NATIVE_EXTENSIONS`` publishes two entries,
// each with its own lane decoder), so a library never holds a kernel it does
// not serve and the family is part of the library's name.
constexpr bool FAMILY_FP8 = TESSERA_ROUTED_FUSED_FP8 != 0;

constexpr int THREADS = 512;
constexpr int PRODUCER_THREADS = 256;
constexpr int BM = 64;                              // routes per superblock
constexpr int BN = 128;                             // B columns per item (two halves)
constexpr int HALF = 64;
constexpr int BK = 32;                              // k columns per chunk
// Column rates the decode reads (bits per code): a column's 512-row chunk is
// ``16 * rate`` int32 words, a 64-row half of it ``2 * rate`` words.  Rates 4
// and 8 put a lane's eight rows on word boundaries; every other rate re-aligns
// the lane's window once per chunk (``decode_rows``).  The run table's rates
// are the grammar's -- one rate, or the two ADJACENT rates bracketing the root
// (``grammar.rate_set``); a non-adjacent pair is refused by name
// (``routed_fused.run_pair``) and traps here -- and every rate 1..8 is read.
// Each (low rate, one or two runs) pair is its own KERNEL instantiation
// (``routed_fused_kernel<..., RL, TWO>``), chosen on the host from the
// launch's ``tile_words`` (``pair_of``): one launch carries one pair, so each
// pair gets its own register allocation and scheduling instead of sharing
// one kernel's with every other pair (``launch_decodes``).
constexpr int RATE_MIN = 1;
constexpr int RATE_MAX = 8;
constexpr int BDESC_INTS = 12;                      // per-32-column descriptor (see ``col_map``)
constexpr int TILE_ROWS = 512;
constexpr int WINDOW_BITS = 14;
constexpr int TABLE_ENTRIES = 1 << WINDOW_BITS;
constexpr int STAGES = 2;
constexpr int WORD_STAGES = 3;

constexpr int TABLE_BYTES = TABLE_ENTRIES * 2;                  // 32768, one table
constexpr int B_STAGE_BYTES = BK * BN * 2;                      // 8192
constexpr int A_STAGE_BYTES = BM * BK * 2;                      // 4096
constexpr int WSCALE_FLOATS = 2 * BN;                           // two item slots
constexpr int DESC_INTS = 2 * 8;
// The shared-memory layout.  The word stages come LAST and are sized at
// launch by the stack's rates: ``Params::slot_words`` int32 words per (half,
// column) slot -- ``slot_words_for_rate`` of the larger rate (2 * rate, plus
// the two words the odd-rate copies start early by), rounded to a multiple of
// 4 so the 16-byte copies stay aligned.  A block on sm_121 may opt in to
// 101,376 B of dynamic shared memory; the two-table gate/up launch needs
// 91,600 + 768 * slot_words, so it fits slots up to 12 words (rates <= 6) and
// not the 16-word slot of rates 7 and 8; the one-table down/dense launch
// (MODE 2) fits every rate.  The host entries check the launch against the
// device's own limit.
//
// The descriptor ring (``OFF_DRING``) holds a two-run chunk's block
// descriptors (``col_map``: BDESC_INTS int32 per projection, one for down and
// dense, two for gate/up) for DRING_STAGES chunks.  They arrive with the word
// stages' copies, two chunks ahead of the words that need them, so a two-run
// chunk maps its columns from shared memory: ``issue_words`` and
// ``load_prev`` each read the global descriptor once per chunk before, and
// that global latency sat on the chunk loop's critical path.  A one-run
// unit's map is computed, so it never touches the ring.
constexpr int DRING_STAGES = 4;
template <int MODE> struct Layout {
    static constexpr int TABLES = (MODE == 2) ? 1 : 2;
    static constexpr int PROJ = (MODE == 2) ? 1 : 2;             // projections per item
    static constexpr int DRING_STAGE = PROJ * BDESC_INTS;        // int32 per ring slot
    static constexpr int OFF_TABLES = 0;
    static constexpr int OFF_B = OFF_TABLES + TABLES * TABLE_BYTES;
    static constexpr int OFF_A = OFF_B + STAGES * B_STAGE_BYTES;
    static constexpr int OFF_WSCALE = OFF_A + STAGES * A_STAGE_BYTES;
    static constexpr int OFF_DESC = OFF_WSCALE + WSCALE_FLOATS * 4;
    static constexpr int OFF_CLAIM = OFF_DESC + DESC_INTS * 4;
    static constexpr int OFF_DRING = OFF_CLAIM + 16;
    static constexpr int OFF_W = OFF_DRING + DRING_STAGES * DRING_STAGE * 4;   // 91,600 (two tables) / 58,640 (one)
    static_assert(OFF_W % 16 == 0, "the word stages take 16-byte copies");
    static_assert(OFF_DRING % 16 == 0, "the descriptor ring takes 16-byte copies");
};
constexpr int SLOT_WORDS_MAX = 2 * RATE_MAX;                    // 16: the rate-8 slot
__host__ __device__ constexpr int w_stage_ints(int slot_words) { return 2 * BK * slot_words; }
__host__ __device__ constexpr int smem_bytes(int mode, int slot_words) {
    return (mode == 2 ? Layout<2>::OFF_W : Layout<0>::OFF_W) + WORD_STAGES * w_stage_ints(slot_words) * 4;
}
// The slot one column at ``rate`` needs: its 2 * rate words, plus two at an
// odd rate -- a 64-row half at an odd rate is 8 * rate bytes at an
// 8 * rate * t64-byte offset, 16-byte aligned only for even t64, and the
// producer copies the odd halves in 16-byte pieces from the aligned word pair
// before them (``issue_words``), so the slot holds those two words too.  The
// decode reads no word past the half (``decode_rows`` loads the next word
// only where a field reaches into it).
__host__ __device__ constexpr int slot_words_for_rate(int rate) {
    return 2 * rate + ((rate & 1) ? 2 : 0);
}
// A launch's slot: the larger of its pair's slots (``r_lo`` and, with a second
// run, ``r_lo + 1``), rounded up to a multiple of 4, at least 4 -- the host's
// ``routed_fused.slot_words_for_pair``.
__host__ __device__ constexpr int pair_slot_words(int r_lo, bool two) {
    const int lo = slot_words_for_rate(r_lo);
    const int hi = two ? slot_words_for_rate(r_lo + 1) : 0;
    const int need = (lo > hi ? lo : hi) > 4 ? (lo > hi ? lo : hi) : 4;
    return (need + 3) / 4 * 4;
}
// The per-block dynamic shared memory sm_121 (GB10, the contract's target)
// lets a kernel opt in to.  It bounds which pairs are INSTANTIATED only; every
// launch is checked against the live device's own limit (``check_slot``), and
// ``routed_fused.SM121_MAX_DYNAMIC_SMEM`` is the same figure.
constexpr int SM121_SMEM_OPTIN = 101376;
// Whether the launch of ``mode`` decodes the pair (``r_lo``; ``two``: a second
// run at ``r_lo + 1``): rates in 1..8, and the pair's slot fits the target's
// block.  The two-table gate/up launch reaches rates 1..6 (one run) and pairs
// up to (5, 6); the one-table down/dense launch every rate and pair.  Only
// these pairs are instantiated; the host refuses any other before a launch
// (``launch``), as ``routed_fused.fused_routed_window_supported`` does first.
__host__ __device__ constexpr bool launch_decodes(int mode, int r_lo, bool two) {
    return r_lo >= RATE_MIN && r_lo + (two ? 1 : 0) <= RATE_MAX
        && smem_bytes(mode, pair_slot_words(r_lo, two)) <= SM121_SMEM_OPTIN;
}
// The run pair a launch's ``tile_words`` fixes.  A 512-row tile holds 16 words
// per unit of column rate, so ``tile_words / 16`` is the sum of the column
// rates, ``K * r_lo + n_hi`` for a pair (r_lo; n_hi columns at r_lo + 1) with
// 0 <= n_hi < K: ``r_lo`` is the quotient and ``n_hi`` the remainder.  (Two
// runs need n_lo, n_hi > 0 -- ``routed_fused.run_pair`` -- and a pair's two
// rates are adjacent, so no other pair has the same sum.)  The kernel checks
// every expert's run table against the pair it was built for and traps on a
// mismatch.
struct PairKey { int r_lo; bool two; };
__host__ __device__ constexpr PairKey pair_of(int tile_words, int K) {
    return PairKey{tile_words / 16 / K, (tile_words / 16) % K != 0};
}
// The largest one-run rate the gate/up launch decodes: published as the
// library's GATE_UP_RATE_MAX, checked at load against
// ``max(routed_fused.ROUTED_LANE_RATES)``, so the rates the host admits on the
// routed-expert launch and the pairs this library instantiates cannot drift.
__host__ __device__ constexpr int gate_up_rate_max() {
    int r = RATE_MIN - 1;
    for (int x = RATE_MIN; x <= RATE_MAX; ++x)
        if (launch_decodes(0, x, false)) r = x;
    return r;
}

constexpr int BAR_FULL0 = 1;
constexpr int BAR_EMPTY0 = 3;
constexpr int BAR_PROD = 5;

__device__ __forceinline__ void bar_sync(int id, int count) {
    asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ void bar_arrive(int id, int count) {
    asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(gmem) : "memory");
}
// The 8-byte copy is the tail of an odd-rate half whose first word is
// 16-byte aligned (t64 even): 2 * rate words is 2 mod 4 there.
__device__ __forceinline__ void cp_async8(void* smem, const void* gmem) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory");
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* smem) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], const void* smem) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
template <bool F16>
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4],
                                         const uint32_t (&b)[2]) {
    if constexpr (F16) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
}
__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint16_t v) {
    uint32_t r;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(v));
    return r;
}
__device__ __forceinline__ uint16_t bf16_bits_rn(float f) {
    return __bfloat16_as_ushort(__float2bfloat16_rn(f));
}
__device__ __forceinline__ float bf16_bits_to_f32(uint32_t bits) {
    return __uint_as_float(bits << 16);
}
// The B tile is [k][n] 16-bit with n contiguous (16 chunks of 16 B per k row).
// The swizzle keeps ldmatrix's eight k-rows of one logical chunk on eight bank
// groups and the decoder's eight chunks of one k-row on eight bank groups too.
__device__ __forceinline__ int bswz(int chunk, int k) {
    return (chunk & 8) | ((chunk ^ (k & 7) ^ ((chunk >> 3) << 1)) & 7);
}
// The A tile is [row][k] 16-bit with k contiguous (4 chunks of 16 B per row).
__device__ __forceinline__ int aswz(int chunk, int row) {
    return chunk ^ ((row >> 1) & 3);
}

// The run table, as the kernel reads it.  The wire's columns are sorted by
// (rate, column) into one or two contiguous runs of the PERMUTED column order
// (``kernel_window_gemv.repack_window_body``); the kernel keeps A and B in
// ORIGINAL column order and maps each original column to its run and its rank
// within the run.  ``runs`` is int32 [E, 8]: (r_lo, 0, n_lo, 0, r_hi, n_lo,
// n_hi, w_hi) with n_hi = 0 when the unit is one run.  ``bdesc`` is int32
// [E, K / 32, BDESC_INTS] per 32-column block: words 0..7 hold 32 bytes, the
// original in-block position of the block's low-rate columns in order, then
// of its high-rate columns; word 8 is the low-rate column count before the
// block, word 9 the block's low-rate column count.  Lane group ``m`` of a
// chunk decodes the block's m-th column in THAT order, so a warp's four
// columns share a rate except in the one warp straddling the two.
struct RunPair { int r_lo, n_lo, r_hi, n_hi, w_hi; };
struct ColMap {
    int rate;   // the column's rate: RL, or RL + 1 in a two-run unit's high run
    bool lo;    // the column is in the low-rate run (always, in a one-run unit)
    int cib;    // its original position within the block: the B-tile column
    int p;      // its permuted index (the run table's column order): init[p]
    int cw0;    // the first word of its chunk within a 512-row tile
};
__device__ __forceinline__ RunPair load_runs(const int32_t* runs, int e) {
    const int4 a = *reinterpret_cast<const int4*>(runs + (long)e * 8);
    const int4 b = *reinterpret_cast<const int4*>(runs + (long)e * 8 + 4);
    RunPair r;
    r.r_lo = a.x; r.n_lo = a.z; r.r_hi = b.x; r.n_hi = b.z; r.w_hi = b.w;
    return r;
}
// Column ``m`` of chunk ``kc`` for the pair (RL; TWO: a second run at RL + 1)
// with ``n_lo`` low-rate columns whose words end at ``w_hi``.
// ``blk`` is chunk kc's block descriptor, in global memory or in the
// descriptor ring (see Layout); a one-run unit never reads it.
template <int RL, bool TWO>
__device__ __forceinline__ ColMap col_map(const int32_t* blk, int n_lo, int w_hi, int kc, int m) {
    ColMap c;
    if constexpr (!TWO) {
        // A one-run unit (every q256 whose root is an integer rate, the
        // v42/v43 rate-4 stacks among them) has the identity descriptor --
        // block_desc lists all 32 columns as low-rate in in-block order, with
        // counts (kc * BK, BK) -- so its map is computed, not read, and with
        // the rate a compile-time constant it folds into the addressing.
        c.rate = RL;
        c.lo = true;
        c.cib = m;
        c.p = kc * BK + m;
        c.cw0 = c.p * 16 * RL;
    } else {
        const int cib = (blk[m >> 2] >> (8 * (m & 3))) & 0xFF;
        const int2 counts = *reinterpret_cast<const int2*>(blk + 8);   // (n_lo_before, cnt_lo)
        c.lo = m < counts.y;
        const int rank = c.lo ? counts.x + m : (kc * BK - counts.x) + (m - counts.y);
        c.rate = c.lo ? RL : RL + 1;
        c.cib = cib;
        c.p = c.lo ? rank : n_lo + rank;
        c.cw0 = c.lo ? rank * 16 * RL : w_hi + rank * 16 * (RL + 1);
    }
    return c;
}
// The words of one column's 64-row half at rate R into its word-stage slot, as
// 16-byte copies split between the two threads ``q`` of the column.  An odd
// rate's half starts on an 8-byte boundary when its index ``t64`` is odd (a
// column's words start 16-byte aligned; 8 * R * t64 is 8 mod 16 there): the
// copies then start at the aligned word pair before the half, which lands in
// the slot's first two words, and the decode reads the half from word 2
// (``odd_off``).  At even t64 the odd rate's last two words are an 8-byte tail.
template <int R>
__device__ __forceinline__ void copy_half(int32_t* dst, const int32_t* src, int t64, int q) {
    if constexpr ((R & 1) == 0) {
        constexpr int N16 = R / 2;                  // 2R words, 16-byte aligned
        #pragma unroll
        for (int k = 0; k < N16; k += 2)
            if (k + q < N16) cp_async16(dst + 4 * (k + q), src + 4 * (k + q));
    } else {
        if (t64 & 1) {
            constexpr int N16 = (R + 1) / 2;        // 2R + 2 words from the pair before the half
            const int32_t* s = src - 2;
            #pragma unroll
            for (int k = 0; k < N16; k += 2)
                if (k + q < N16) cp_async16(dst + 4 * (k + q), s + 4 * (k + q));
        } else {
            constexpr int N16 = (R - 1) / 2;        // 2R = 4 * N16 + 2 words: an 8-byte tail
            #pragma unroll
            for (int k = 0; k < N16; k += 2)
                if (k + q < N16) cp_async16(dst + 4 * (k + q), src + 4 * (k + q));
            if (q == (N16 & 1)) cp_async8(dst + 4 * N16, src + 4 * N16);
        }
    }
}

// Eight rows of one column of one half, from the half's word slot: the
// window state of row ``n`` is the last 14 bits of the column's MSB-first
// stream ending after row ``n``.  Lane ``j`` holds rows 8j..8j+7, whose bits
// start at stream bit ``8 * j * R`` of the half: word ``b`` of the slot, bit
// ``u`` into it.  The lane re-aligns a three-word window on ``u`` once
// (``Z0 | Z1 | Z2`` = the 96 stream bits from 32 before its first row), after
// which every row's field sits at a compile-time position; at rates 4 and 8
// ``u`` is 0 and the window is the slot's words themselves -- the rate-4 path
// is exactly the original kernel's constant shifts.  ``prev`` is the word
// before the half's first word (the previous 64 rows, the previous tile's
// last word of the column, the cut's start state, or zero).
template <bool FP8, int R>
__device__ __forceinline__ void decode_rows(const int32_t* Wc, uint32_t prev, int j,
                                            const uint16_t* T, const float* ws,
                                            uint32_t (&packed)[4]) {
    constexpr bool ALIGNED = (8 * R) % 32 == 0;
    constexpr bool WIDE = 8 * R > 32;                 // rows reach past 32 bits: Z2 is read
    const int bits0 = 8 * j * R;
    const int b = bits0 >> 5;
    const uint32_t wm1 = (b > 0) ? (uint32_t)Wc[b - 1] : prev;
    const uint32_t w0 = (uint32_t)Wc[b];
    uint32_t Z0, Z1, Z2 = 0;
    if constexpr (ALIGNED) {
        Z0 = wm1;
        Z1 = w0;
        if constexpr (WIDE) Z2 = (uint32_t)Wc[b + 1];
    } else {
        const int u = bits0 & 31;                     // 1..31 here, 0 only for j = 0
        // The lane's eight fields end 8 * R bits after bits0; the next word is
        // read only where a field reaches into it, so the last lane of a half
        // never reads past the half's 2 * R words.
        const uint32_t w1 = (u + 8 * R > 32) ? (uint32_t)Wc[b + 1] : 0u;
        // the 32 stream bits starting u into (hi:lo), MSB-first; u = 0 gives hi
        Z0 = __funnelshift_rc(w0, wm1, 32 - u);
        Z1 = __funnelshift_rc(w1, w0, 32 - u);
        if constexpr (WIDE) {
            const uint32_t w2 = (u + 8 * R > 64) ? (uint32_t)Wc[b + 2] : 0u;
            Z2 = __funnelshift_rc(w2, w1, 32 - u);
        }
    }
    #pragma unroll
    for (int r = 0; r < 8; r += 2) {
        uint32_t s[2];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int e = (r + i + 1) * R;            // the field ends here, in the window
            const int k1 = (e - 1) >> 5;              // 0: (Z0:Z1), 1: (Z1:Z2)
            const int shift = 32 * (k1 + 1) - e;
            const uint32_t lo = k1 ? Z2 : Z1;
            const uint32_t hi = k1 ? Z1 : Z0;
            s[i] = __funnelshift_r(lo, hi, shift) & 0x3FFFu;
        }
        uint32_t t0 = T[s[0]], t1 = T[s[1]];
        if constexpr (!FP8) {
            // FOLDED: one bf16 rounding of value * row_scale, before the dot
            t0 = bf16_bits_rn(__fmul_rn(bf16_bits_to_f32(t0), ws[r]));
            t1 = bf16_bits_rn(__fmul_rn(bf16_bits_to_f32(t1), ws[r + 1]));
        }
        packed[r >> 1] = t0 | (t1 << 16);
    }
}
// Both halves of a chunk (rates RA and RB, each compile-time) as ONE
// straight-line block, so the scheduler can issue the second half's word and
// table loads while the first half's are in flight.
template <bool FP8, int RA, int RB>
__device__ __forceinline__ void decode_two(const int32_t* const (&Wc)[2], const int32_t (&prev)[2], int j,
                                           const uint16_t* T0, const uint16_t* T1,
                                           const float* const (&ws)[2], uint32_t (&packed)[2][4]) {
    decode_rows<FP8, RA>(Wc[0], (uint32_t)prev[0], j, T0, ws[0], packed[0]);
    decode_rows<FP8, RB>(Wc[1], (uint32_t)prev[1], j, T1, ws[1], packed[1]);
}

struct Params {
    const void* x;                 // [rows_x, K] bf16 (value) or e4m3 (fp8)
    const float* a_scale;          // [rows_x] fp32 (fp8) or nullptr
    const int32_t* words0;         // [E, words_stride] gate (mode 0/1) or down (mode 2)
    const int32_t* words1;         // [E, words_stride] up (mode 0/1) or nullptr
    const uint16_t* table0;        // [E, 16384]
    const uint16_t* table1;
    const int32_t* init0;          // [E, K]
    const int32_t* init1;
    const int32_t* has_init0;      // [E]
    const int32_t* has_init1;
    const float* wscale0;          // [E, N]
    const float* wscale1;
    const int32_t* runs0;          // [E, 8] run pairs (see RunPair)
    const int32_t* runs1;
    const int32_t* bdesc0;         // [E, K / 32, BDESC_INTS] block descriptors (see col_map)
    const int32_t* bdesc1;
    long words_stride;
    int tile_words;
    int slot_words;                // int32 words per (half, column) word-stage slot (see Layout)
    int K;
    int N;                         // rows per projection
    int E;
    const int32_t* offsets;        // [E + 1]
    const int32_t* flat_sorted;    // [P]
    const float* rw_sorted;        // [P]
    const int32_t* item_off;       // [E + 1] cumulative superblocks
    int32_t* counter;
    int n_blocks;                  // items per superblock
    int top_k;
    int a_row_mode;                // 0 token = flat / top_k, 1 sorted position, 2 flat
    int mul_weight;
    float limit;
    void* out;                     // bf16
    long out_stride;
    int inter;                     // I (mode 1: the up half's column offset)
    int rows_x;                    // DENSE: M, the rows of x (and of out)
    int k_split;                   // DENSE: S, the K-range splits per (n-block, superblock)
    float* partial;                // DENSE && SPLIT: fp32 [S, M, N] raw accumulators
};

// ``RL``, ``TWO``: the launch's run pair (``pair_of``) -- the low (or only)
// rate, and whether a second run at ``RL + 1`` exists.
template <bool FP8, int MODE, bool DENSE, bool SPLIT, int RL, bool TWO>
__global__ void __launch_bounds__(THREADS, 1) routed_fused_kernel(const Params p) {
    static_assert(!DENSE || MODE == 2, "the dense case is the single-projection (down) mode");
    static_assert(!SPLIT || DENSE, "a K split is a dense scheduling device");
    static_assert(launch_decodes(MODE, RL, TWO), "only the pairs the launch decodes are instantiated");
    using L = Layout<MODE>;
    extern __shared__ __align__(128) uint8_t smem[];
    uint16_t* tab = reinterpret_cast<uint16_t*>(smem + L::OFF_TABLES);
    uint8_t* Bs = smem + L::OFF_B;
    uint8_t* As = smem + L::OFF_A;
    int32_t* Ws = reinterpret_cast<int32_t*>(smem + L::OFF_W);
    float* wsc = reinterpret_cast<float*>(smem + L::OFF_WSCALE);
    int32_t* desc = reinterpret_cast<int32_t*>(smem + L::OFF_DESC);
    int32_t* claim = reinterpret_cast<int32_t*>(smem + L::OFF_CLAIM);

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int nk = p.K / BK;
    const int dense_nsb = (p.rows_x + BM - 1) / BM;      // DENSE: superblocks of x
    const int total_items = DENSE ? dense_nsb * p.k_split * p.n_blocks
                                  : p.item_off[p.E] * p.n_blocks;
    unsigned gc = 0;          // global chunk counter: the stage is gc & 1
    unsigned item_idx = 0;    // the descriptor slot is item_idx & 1

    if (tid < PRODUCER_THREADS) {
        // ------------------------------------------------------------ producers
        const int m = tid >> 3;     // lane group: the chunk's m-th column in the block's
                                    // (low-rate, high-rate) order -- see col_map
        const int j = tid & 7;      // eight rows (8j..8j+7) within the half
        int last_e = -1;            // the expert whose table(s) shared memory holds
        for (;;) {
            bar_sync(BAR_PROD, PRODUCER_THREADS);   // every producer is done with the last item's smem
            if (tid == 0) claim[0] = atomicAdd(p.counter, 1);
            bar_sync(BAR_PROD, PRODUCER_THREADS);
            const int item = claim[0];
            const int slot = item_idx & 1;
            if (item >= total_items) {
                if (tid == 0) desc[slot * 8 + 0] = -1;
                __threadfence_block();
                bar_arrive(BAR_FULL0 + (gc & 1), THREADS);
                return;
            }
            // item -> (expert, n-block, superblock); items of one expert are
            // contiguous and ordered (n-block, superblock) so that neighbouring
            // items read the same words.  DENSE: one expert, the superblocks
            // of x in order, and a K-split index innermost so the S splits of
            // one (n-block, superblock) run on neighbouring SMs.
            int e, nb, sb, pos0, mb, kc0, nkc, ks;
            if constexpr (DENSE) {
                const int per_nb = dense_nsb * p.k_split;
                nb = item / per_nb;
                const int rem = item - nb * per_nb;
                sb = rem / p.k_split;
                ks = rem - sb * p.k_split;
                e = 0;
                pos0 = sb * BM;
                mb = min(BM, p.rows_x - pos0);
                kc0 = (int)(((long)ks * nk) / p.k_split);
                nkc = (int)(((long)(ks + 1) * nk) / p.k_split) - kc0;
            } else {
                const int sbg = item / p.n_blocks;
                int lo = 0, hi = p.E;
                while (hi - lo > 1) {
                    const int mid = (lo + hi) >> 1;
                    if (p.item_off[mid] <= sbg) lo = mid; else hi = mid;
                }
                e = lo;
                const int nsb = p.item_off[e + 1] - p.item_off[e];
                const int local = item - p.item_off[e] * p.n_blocks;
                nb = local / nsb;
                sb = local - nb * nsb;
                const int start = p.offsets[e];
                const int end = p.offsets[e + 1];
                pos0 = start + sb * BM;
                mb = min(BM, end - pos0);
                kc0 = 0;
                nkc = nk;
                ks = 0;
            }
            if (tid == 0) {
                desc[slot * 8 + 0] = e;
                desc[slot * 8 + 1] = nb;
                desc[slot * 8 + 2] = sb;
                desc[slot * 8 + 3] = pos0;
                desc[slot * 8 + 4] = mb;
                desc[slot * 8 + 5] = kc0;
                desc[slot * 8 + 6] = nkc;
                desc[slot * 8 + 7] = ks;
            }
            const int n0 = (MODE == 2) ? nb * BN : nb * HALF;
            // Row scales for the item's 128 B columns, in B-column order.
            if (tid < BN) {
                const int c = tid;
                float v;
                if (MODE == 2) {
                    v = p.wscale0[(long)e * p.N + n0 + c];
                } else {
                    const int q = c >> 5, r = c & 31;
                    const int h = r >> 4;
                    const float* ws = h ? p.wscale1 : p.wscale0;
                    v = ws[(long)e * p.N + n0 + q * 16 + (r & 15)];
                }
                wsc[slot * BN + c] = v;
            }
            // The expert's table(s): 32 KB each, asynchronously -- unless shared
            // memory already holds this expert's, which it does for every item
            // after the first of one expert (always, in the dense case).  Safe
            // because every producer passed the barrier above after its last
            // lookup of the previous item, and consumers never read the table.
            if (e != last_e) {
                const uint16_t* t0 = p.table0 + (long)e * TABLE_ENTRIES;
                for (int i = tid; i < TABLE_ENTRIES / 8; i += PRODUCER_THREADS)
                    cp_async16(tab + i * 8, t0 + i * 8);
                if (MODE != 2) {
                    const uint16_t* t1 = p.table1 + (long)e * TABLE_ENTRIES;
                    for (int i = tid; i < TABLE_ENTRIES / 8; i += PRODUCER_THREADS)
                        cp_async16(tab + TABLE_ENTRIES + i * 8, t1 + i * 8);
                }
                last_e = e;
            }
            // Word geometry per half: half h decodes rows n_h0 .. n_h0 + 63 of
            // its projection (down: the two halves of one 128-row block; gate/up:
            // the same 64 intermediate rows of two projections).  A column's
            // words for the half start ``2 * rate * t64`` words into its chunk
            // of tile g (``t64`` = the half's 64-row index within the tile).
            const int32_t* tbase_h[2];
            int g_h[2], t64_h[2];
            const int32_t* init_h[2];
            int hasinit_h[2];
            RunPair rp_h[2];
            const int32_t* bdesc_h[2];
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int nh0 = (MODE == 2) ? n0 + HALF * h : n0;
                const int g = nh0 / TILE_ROWS;
                const int t = nh0 - g * TILE_ROWS;
                const bool second = (MODE != 2) && h == 1;
                g_h[h] = g;
                t64_h[h] = t / HALF;
                tbase_h[h] = (second ? p.words1 : p.words0) + (long)e * p.words_stride
                             + (long)g * p.tile_words;
                init_h[h] = (second ? p.init1 : p.init0) + (long)e * p.K;
                hasinit_h[h] = (second ? p.has_init1 : p.has_init0)[e];
                rp_h[h] = load_runs(second ? p.runs1 : p.runs0, e);
                bdesc_h[h] = (second ? p.bdesc1 : p.bdesc0) + (long)e * nk * BDESC_INTS;
            }
            // The run pair must tile K and the wire's tile_words exactly, its
            // two rates must be adjacent (``grammar.rate_set``: a stack mixes
            // only the two bracketing its root; ``run_pair`` refuses any other
            // pair by name), and gate and up must carry ONE pair: the one this
            // kernel is instantiated for.  The Python owner checks the first
            // two per stack; the third needs no check of its own: equal
            // ``cols`` and equal ``tile_words`` (``fused_routed_window_supported``)
            // fix 16 * (K * r_lo + n_hi), with 0 <= n_hi < K, and so the whole
            // adjacent pair -- the one the host chose this instantiation from
            // (``pair_of``).  A mismatch here would address outside the
            // expert's words, so it traps rather than reads.
            if (tid == 0) {
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const RunPair& rp = rp_h[h];
                    if (rp.r_lo != RL || (rp.n_hi > 0) != TWO
                        || rp.n_lo + rp.n_hi != p.K
                        || 16 * (rp.n_lo * rp.r_lo + rp.n_hi * rp.r_hi) != p.tile_words
                        || rp.r_lo < RATE_MIN || rp.r_lo > RATE_MAX
                        || slot_words_for_rate(rp.r_lo) > p.slot_words
                        || (rp.n_hi > 0 && (rp.r_hi != rp.r_lo + 1 || rp.r_hi > RATE_MAX
                                            || rp.w_hi != 16 * rp.n_lo * rp.r_lo
                                            || slot_words_for_rate(rp.r_hi) > p.slot_words)))
                        __trap();
                }
                if (rp_h[1].r_lo != rp_h[0].r_lo || rp_h[1].n_lo != rp_h[0].n_lo || rp_h[1].n_hi != rp_h[0].n_hi)
                    __trap();
            }
            // The item's pair (both halves', checked above).
            const int n_lo = rp_h[0].n_lo;
            const int w_hi = rp_h[0].w_hi;
            // The A row this thread stages (-1: a zero row past the superblock).
            long arow = -1;
            {
                const int r = FP8 ? (tid >> 1) : (tid >> 2);
                if ((!FP8 || tid < 128) && r < mb) {
                    const int pos = pos0 + r;
                    if constexpr (DENSE) {
                        arow = pos;                       // row m of x is route m
                    } else {
                        const int flat = p.flat_sorted[pos];
                        arow = (p.a_row_mode == 1) ? pos : (p.a_row_mode == 0 ? flat / p.top_k : flat);
                    }
                }
            }
            // The half this thread issues words for (threads 0..127; fixed per
            // item) -- selected once, so the per-half tables stay in registers
            // instead of becoming a runtime-indexed local array.
            const int ih = (tid >> 6) & 1;
            const int32_t* tbase_i = ih ? tbase_h[1] : tbase_h[0];
            const int t64_i = ih ? t64_h[1] : t64_h[0];
            auto load_a = [&](int kc, uint4& a) {
                if (arow >= 0) {
                    if constexpr (FP8) {
                        const uint8_t* src = reinterpret_cast<const uint8_t*>(p.x) + arow * p.K + kc * BK + (tid & 1) * 16;
                        a = *reinterpret_cast<const uint4*>(src);
                    } else {
                        const uint16_t* src = reinterpret_cast<const uint16_t*>(p.x) + arow * p.K + kc * BK + (tid & 3) * 8;
                        a = *reinterpret_cast<const uint4*>(src);
                    }
                }
            };
            auto store_a = [&](int stage, const uint4& a) {
                uint8_t* A = As + stage * A_STAGE_BYTES;
                if constexpr (FP8) {
                    if (tid >= 128) return;
                    const int row = tid >> 1;
                    const int c16 = tid & 1;
                    uint4 lo, hi;
                    if (arow >= 0) {
                        lo.x = e4m3x2_to_f16x2((uint16_t)(a.x & 0xFFFF)); lo.y = e4m3x2_to_f16x2((uint16_t)(a.x >> 16));
                        lo.z = e4m3x2_to_f16x2((uint16_t)(a.y & 0xFFFF)); lo.w = e4m3x2_to_f16x2((uint16_t)(a.y >> 16));
                        hi.x = e4m3x2_to_f16x2((uint16_t)(a.z & 0xFFFF)); hi.y = e4m3x2_to_f16x2((uint16_t)(a.z >> 16));
                        hi.z = e4m3x2_to_f16x2((uint16_t)(a.w & 0xFFFF)); hi.w = e4m3x2_to_f16x2((uint16_t)(a.w >> 16));
                    } else {
                        lo = make_uint4(0, 0, 0, 0); hi = lo;
                    }
                    *reinterpret_cast<uint4*>(A + row * (BK * 2) + (aswz(2 * c16, row) << 4)) = lo;
                    *reinterpret_cast<uint4*>(A + row * (BK * 2) + (aswz(2 * c16 + 1, row) << 4)) = hi;
                } else {
                    const int row = tid >> 2;
                    const int c = tid & 3;
                    const uint4 v = (arow >= 0) ? a : make_uint4(0, 0, 0, 0);
                    *reinterpret_cast<uint4*>(A + row * (BK * 2) + (aswz(c, row) << 4)) = v;
                }
            };

            // The item's chunk loop, for the kernel's run pair: ``RL`` the low
            // (or only) rate, ``TWO`` whether a second run at ``RL + 1``
            // exists.  The slot size, the copy pattern, the window shifts and --
            // for one run -- the column map are compile-time, so a uniform
            // stack decodes with the constant shifts and addressing of the v42
            // rate-4 kernel, and both halves' decodes are one straight-line
            // block (``decode_two``).  tessera#694: the first version switched
            // on the column's rate per half per chunk, and its rate-4 launches
            // ran 1.31x (down) to 1.49x (gate/up) slower than the v44 kernel's.
            // Until the pair became a kernel template parameter, one kernel
            // held every pair's loop behind a per-item switch, and a change to
            // the two-run loop moved the one-run loop's registers: the
            // column-map ring cost the rate-4 gate/up launch 11% more
            // instructions on its table lookups with its own source unchanged
            // (docs/measurements/2026-09-29-two-run-column-map.md).
            auto run = [&]() {
                constexpr int RH = RL + 1;                  // read only when TWO
                constexpr int SW = pair_slot_words(RL, TWO);
                constexpr int W_STAGE = w_stage_ints(SW);
                // Chunk kc's block descriptor for half h: the expert's, in
                // global memory (``ring`` false: the first chunks, before the
                // ring's copies land), or its copy in the descriptor ring (see
                // Layout).  The ring's address is formed here, where it is
                // used, not held in a register for the whole chunk loop.
                auto blk_of = [&](int kc, int h, bool ring) -> const int32_t* {
                    if (ring)
                        return reinterpret_cast<const int32_t*>(smem + L::OFF_DRING)
                               + (kc % DRING_STAGES) * L::DRING_STAGE + ((MODE == 2) ? 0 : h) * BDESC_INTS;
                    return ((MODE != 2 && h) ? bdesc_h[1] : bdesc_h[0]) + (long)kc * BDESC_INTS;
                };
                // Chunk kc's descriptors into the ring: 16 bytes per thread,
                // from producer threads that issue no words.  A two-run unit
                // only (a one-run unit's map is computed).
                auto issue_desc = [&](int kc) {
                    if constexpr (TWO) {
                        const int t = tid - 128;
                        if (t >= 0 && t < 3 * L::PROJ) {
                            const int h = t >= 3, part = t - 3 * h;
                            const int32_t* src = (h ? bdesc_h[1] : bdesc_h[0]) + (long)kc * BDESC_INTS + 4 * part;
                            cp_async16(reinterpret_cast<int32_t*>(smem + L::OFF_DRING)
                                           + (kc % DRING_STAGES) * L::DRING_STAGE + h * BDESC_INTS + 4 * part, src);
                        }
                    }
                };
                // The words of chunk kc for half ih, lane group mm (``copy_half``).
                // SW never exceeds the launch's slot (the trap check above), so
                // the stages fit the shared memory the host sized.
                auto issue_words = [&](int kc, bool ring) {
                    if (tid < 128) {
                        const int mm = (tid >> 1) & 31;
                        const int q = tid & 1;
                        const ColMap c = col_map<RL, TWO>(blk_of(kc, ih, ring), n_lo, w_hi, kc, mm);
                        const int32_t* src = tbase_i + c.cw0 + 2 * c.rate * t64_i;
                        int32_t* dst = Ws + (kc % WORD_STAGES) * W_STAGE + (ih * BK + mm) * SW;
                        if constexpr (TWO) {
                            if (c.lo) copy_half<RL>(dst, src, t64_i, q);
                            else copy_half<RH>(dst, src, t64_i, q);
                        } else {
                            copy_half<RL>(dst, src, t64_i, q);
                        }
                    }
                };
                // The 32 stream bits before the half's first word, for every row
                // group whose window starts inside that word (8 * j * rate < 32:
                // j = 0 at any rate, j <= 3 at rate 1, j = 1 at rates 2 and 3) --
                // a field's 14-bit window reaches up to 13 bits before it, and at
                // rate 1 the second group's does reach past the word (tessera#694:
                // loading it for j = 0 alone left rows 8..12 of every half after
                // the first reading a zero history at rate 1).  The chunk's column
                // map comes back too: the chunk's decode, one iteration later,
                // reads it instead of mapping the column again.  A two-run
                // chunk maps from the descriptor ring (``ring``; see its
                // schedule below).
                auto load_prev = [&](int kc, int32_t (&pv)[2], ColMap (&cm)[2], bool ring) {
                    #pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        // MODE 2 reads one projection: both halves map alike.
                        cm[h] = (MODE == 2 && h == 1) ? cm[0] : col_map<RL, TWO>(blk_of(kc, h, ring), n_lo, w_hi, kc, m);
                        const ColMap& c = cm[h];
                        if (8 * j * c.rate >= 32) continue;
                        const int wr0 = 2 * c.rate * t64_h[h];
                        const int32_t* wcol = tbase_h[h] + c.cw0;
                        int32_t v;
                        if (wr0 > 0) v = wcol[wr0 - 1];
                        else if (g_h[h] > 0) v = wcol[16 * c.rate - 1 - p.tile_words];
                        else v = hasinit_h[h] ? init_h[h][c.p] : 0;
                        pv[h] = v;
                    }
                };

                // The descriptor ring's schedule.  Chunks kc0 and kc0 + 1's
                // descriptors are stored directly (read from global memory now;
                // a producer barrier makes them visible), chunk kc0 + 2's copy
                // joins group 0 (the tables and chunk kc0's words), chunk
                // kc0 + 3's group 1 (chunk kc0 + 1's words), and the group
                // iteration kc commits carries chunk kc + 2's words and chunk
                // kc + 4's descriptors.  So chunk kc + 1's descriptors are
                // visible to every producer from iteration kc's top (the
                // map for ``load_prev(kc + 1)``) and chunk kc + 2's past that
                // iteration's wait and barrier (``issue_words(kc + 2)``).  The
                // copy of chunk kc + 4 overwrites chunk kc's slot only past
                // iteration kc's barrier, which every reader of chunk kc's
                // descriptors (``issue_words`` at iteration kc - 2,
                // ``load_prev`` at kc - 1) passed after reading them.  The first
                // two chunks' words map from the global descriptors.
                if constexpr (TWO) {
                    const int t = tid - 128;
                    if (t >= 0 && t < 3 * L::PROJ) {
                        const int h = t >= 3, part = t - 3 * h;
                        const int32_t* src = (h ? bdesc_h[1] : bdesc_h[0]) + (long)kc0 * BDESC_INTS + 4 * part;
                        int32_t* dst = reinterpret_cast<int32_t*>(smem + L::OFF_DRING) + h * BDESC_INTS + 4 * part;
                        #pragma unroll
                        for (int c = 0; c < 2; ++c)
                            if (c < nkc)
                                *reinterpret_cast<int4*>(dst + ((kc0 + c) % DRING_STAGES) * L::DRING_STAGE) =
                                    *reinterpret_cast<const int4*>(src + c * BDESC_INTS);
                    }
                }
                issue_words(kc0, false);
                if (nkc > 2) issue_desc(kc0 + 2);
                cp_async_commit();                     // group 0
                if (nkc > 1) issue_words(kc0 + 1, false);
                if (nkc > 3) issue_desc(kc0 + 3);
                cp_async_commit();                     // group 1
                if constexpr (TWO) bar_sync(BAR_PROD, PRODUCER_THREADS);   // chunks kc0, kc0 + 1's descriptors
                int32_t prev_cur[2] = {0, 0}, prev_nxt[2] = {0, 0};
                ColMap cm_cur[2], cm_nxt[2];
                uint4 a_cur = make_uint4(0, 0, 0, 0), a_nxt = make_uint4(0, 0, 0, 0);
                load_prev(kc0, prev_cur, cm_cur, TWO);
                load_a(kc0, a_cur);
                // Settle the first chunk's loads here, before the chunk loop.
                // They land in the registers the loop carries (``prev_cur``,
                // ``a_cur``), so without a use here ptxas guards those
                // registers with the loads' scoreboard on EVERY iteration, and
                // the next chunk's loads (``load_prev``/``load_a`` at the top
                // of the loop) share that scoreboard: the chunk's store and
                // decode then waited for the NEXT chunk's global loads, one
                // global latency per chunk.  XOR with a zero the compiler
                // cannot fold (``K`` is positive) consumes them here, and the
                // loop's only wait on those loads is at its end, where the next
                // chunk's values move into place, behind the chunk's own work.
                // Verified on the image's ptxas (CUDA 13.0.88); re-check the chunk
                // loop's scoreboard waits on a toolchain change.
                {
                    const int32_t zero = p.K >> 31;
                    prev_cur[0] ^= zero; prev_cur[1] ^= zero;
                    a_cur.x ^= (uint32_t)zero; a_cur.y ^= (uint32_t)zero;
                    a_cur.z ^= (uint32_t)zero; a_cur.w ^= (uint32_t)zero;
                }
                for (int ic = 0; ic < nkc; ++ic, ++gc) {
                    const int kc = kc0 + ic;
                    // The two orders are the measured ones: a one-run loop that
                    // issues the activation chunk first waits longer at M = 1.
                    if constexpr (TWO) {
                        if (ic + 1 < nkc) { load_a(kc + 1, a_nxt); load_prev(kc + 1, prev_nxt, cm_nxt, true); }
                    } else {
                        if (ic + 1 < nkc) { load_prev(kc + 1, prev_nxt, cm_nxt, false); load_a(kc + 1, a_nxt); }
                    }
                    cp_async_wait<1>();                // chunk kc's words (and the tables) have landed
                    bar_sync(BAR_PROD, PRODUCER_THREADS);   // ... for every producer; chunk kc-1's stage is free
                    if (ic + 2 < nkc) issue_words(kc + 2, TWO);
                    if (ic + 4 < nkc) issue_desc(kc + 4);
                    cp_async_commit();
                    const int stage = gc & 1;
                    if (gc >= 2) bar_sync(BAR_EMPTY0 + stage, THREADS);
                    store_a(stage, a_cur);
                    const int32_t* W = Ws + (kc % WORD_STAGES) * W_STAGE;
                    uint8_t* B = Bs + stage * B_STAGE_BYTES;
                    const int32_t* Wc[2];
                    const float* ws[2];
                    int chunk[2];
                    #pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int odd_off = ((cm_cur[h].rate & 1) & (t64_h[h] & 1)) << 1;   // see copy_half
                        Wc[h] = W + (h * BK + m) * SW + odd_off;
                        chunk[h] = (MODE == 2) ? (8 * h + j) : (4 * (j >> 1) + 2 * h + (j & 1));
                        ws[h] = wsc + slot * BN + chunk[h] * 8;
                    }
                    const uint16_t* T0 = tab;
                    const uint16_t* T1 = tab + ((MODE == 2) ? 0 : TABLE_ENTRIES);
                    uint32_t packed[2][4];
                    if constexpr (!TWO) {
                        decode_two<FP8, RL, RL>(Wc, prev_cur, j, T0, T1, ws, packed);
                    } else if constexpr (MODE == 2) {
                        // one column in both halves: one rate
                        if (cm_cur[0].lo) decode_two<FP8, RL, RL>(Wc, prev_cur, j, T0, T1, ws, packed);
                        else decode_two<FP8, RH, RH>(Wc, prev_cur, j, T0, T1, ws, packed);
                    } else {
                        // gate and up share the pair, not the column order
                        const bool lo0 = cm_cur[0].lo, lo1 = cm_cur[1].lo;
                        if (lo0 && lo1) decode_two<FP8, RL, RL>(Wc, prev_cur, j, T0, T1, ws, packed);
                        else if (!lo0 && !lo1) decode_two<FP8, RH, RH>(Wc, prev_cur, j, T0, T1, ws, packed);
                        else if (lo0) decode_two<FP8, RL, RH>(Wc, prev_cur, j, T0, T1, ws, packed);
                        else decode_two<FP8, RH, RL>(Wc, prev_cur, j, T0, T1, ws, packed);
                    }
                    #pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int cib = cm_cur[h].cib;
                        *reinterpret_cast<uint4*>(B + cib * (BN * 2) + (bswz(chunk[h], cib) << 4)) =
                            make_uint4(packed[h][0], packed[h][1], packed[h][2], packed[h][3]);
                    }
                    bar_arrive(BAR_FULL0 + stage, THREADS);
                    prev_cur[0] = prev_nxt[0]; prev_cur[1] = prev_nxt[1];
                    cm_cur[0] = cm_nxt[0]; cm_cur[1] = cm_nxt[1];
                    a_cur = a_nxt;
                }
            };
            // Every item of a launch carries the kernel's pair (one run table
            // per stack, gate and up alike; checked above).
            run();
            ++item_idx;
        }
    } else {
        // ------------------------------------------------------------ consumers
        const int cw = (tid - PRODUCER_THREADS) >> 5;
        const int mw = cw >> 2;      // 32 rows
        const int nw = cw & 3;       // 32 B columns
        for (;;) {
            const int slot = item_idx & 1;
            int stage = gc & 1;
            bar_sync(BAR_FULL0 + stage, THREADS);
            const int e = desc[slot * 8 + 0];
            if (e < 0) return;
            const int nb = desc[slot * 8 + 1];
            const int pos0 = desc[slot * 8 + 3];
            const int mb = desc[slot * 8 + 4];
            const int nkc = desc[slot * 8 + 6];
            const int ks = desc[slot * 8 + 7];
            float acc[2][4][4];
            #pragma unroll
            for (int mi = 0; mi < 2; ++mi)
                #pragma unroll
                for (int nt = 0; nt < 4; ++nt)
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) acc[mi][nt][i] = 0.f;
            for (int ic = 0; ic < nkc; ++ic, ++gc) {
                stage = gc & 1;
                if (ic > 0) bar_sync(BAR_FULL0 + stage, THREADS);
                const uint8_t* A = As + stage * A_STAGE_BYTES;
                const uint8_t* B = Bs + stage * B_STAGE_BYTES;
                #pragma unroll
                for (int s = 0; s < BK / 16; ++s) {
                    uint32_t a[2][4];
                    const int q = lane >> 3;
                    #pragma unroll
                    for (int mi = 0; mi < 2; ++mi) {
                        const int row = 32 * mw + 16 * mi + 8 * (q & 1) + (lane & 7);
                        const int kch = 2 * s + (q >> 1);
                        ldmatrix_x4(a[mi], A + row * (BK * 2) + (aswz(kch, row) << 4));
                    }
                    uint32_t b[4][2];
                    #pragma unroll
                    for (int pair = 0; pair < 2; ++pair) {
                        const int rk = 16 * s + 8 * (q & 1) + (lane & 7);
                        const int ch = 4 * nw + 2 * pair + (q >> 1);
                        uint32_t r[4];
                        ldmatrix_x4_trans(r, B + rk * (BN * 2) + (bswz(ch, rk) << 4));
                        b[2 * pair][0] = r[0]; b[2 * pair][1] = r[1];
                        b[2 * pair + 1][0] = r[2]; b[2 * pair + 1][1] = r[3];
                    }
                    #pragma unroll
                    for (int mi = 0; mi < 2; ++mi)
                        #pragma unroll
                        for (int nt = 0; nt < 4; ++nt) mma16816<FP8>(acc[mi][nt], a[mi], b[nt]);
                }
                bar_arrive(BAR_EMPTY0 + stage, THREADS);
            }
            // ------------------------------------------------------ epilogue
            const int n0 = (MODE == 2) ? nb * BN : nb * HALF;
            uint16_t* out = reinterpret_cast<uint16_t*>(p.out);
            #pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                #pragma unroll
                for (int hr = 0; hr < 2; ++hr) {
                    const int r = 32 * mw + 16 * mi + 8 * hr + (lane >> 2);
                    if (r >= mb) continue;
                    const int pos = pos0 + r;
                    const int flat = DENSE ? pos : p.flat_sorted[pos];
                    float a_s = 1.f, rw = 1.f;
                    if constexpr (FP8 && !SPLIT) {
                        const long arow = DENSE ? (long)pos
                            : ((p.a_row_mode == 1) ? pos : (p.a_row_mode == 0 ? flat / p.top_k : flat));
                        a_s = p.a_scale[arow];
                    }
                    if (MODE == 2 && !DENSE && p.mul_weight) rw = p.rw_sorted[pos];
                    if constexpr (SPLIT) {
                        // The raw fp32 accumulator of this K range; the reduce
                        // kernel forms the sum in split order and applies the
                        // epilogue once.
                        float* part = p.partial + ((long)ks * p.rows_x + pos) * p.N + n0;
                        #pragma unroll
                        for (int nt = 0; nt < 4; ++nt) {
                            const int cb = 32 * nw + 8 * nt + 2 * (lane & 3);
                            *reinterpret_cast<float2*>(part + cb) =
                                make_float2(acc[mi][nt][2 * hr], acc[mi][nt][2 * hr + 1]);
                        }
                    } else if constexpr (MODE == 0) {
                        // gate n-tiles 0,1 pair with up n-tiles 2,3 in the same registers
                        #pragma unroll
                        for (int nt = 0; nt < 2; ++nt) {
                            const int cb = 32 * nw + 8 * nt + 2 * (lane & 3);      // gate B column
                            uint32_t two = 0;
                            #pragma unroll
                            for (int i = 0; i < 2; ++i) {
                                float g = acc[mi][nt][2 * hr + i];
                                float u = acc[mi][nt + 2][2 * hr + i];
                                if constexpr (FP8) {
                                    g = __fmul_rn(__fmul_rn(g, a_s), wsc[slot * BN + cb + i]);
                                    u = __fmul_rn(__fmul_rn(u, a_s), wsc[slot * BN + cb + 16 + i]);
                                }
                                // the bf16 GEMM output, widened for the fp32 activation
                                float gf = bf16_bits_to_f32(bf16_bits_rn(g));
                                float uf = bf16_bits_to_f32(bf16_bits_rn(u));
                                gf = fminf(gf, p.limit);
                                uf = fmaxf(fminf(uf, p.limit), -p.limit);
                                const float act = __fmul_rn(gf / (1.0f + expf(-gf)), uf);
                                two |= (uint32_t)bf16_bits_rn(act) << (16 * i);
                            }
                            const long col = n0 + 16 * nw + 8 * nt + 2 * (lane & 3);
                            *reinterpret_cast<uint32_t*>(out + (long)pos * p.out_stride + col) = two;
                        }
                    } else if constexpr (MODE == 1) {
                        #pragma unroll
                        for (int nt = 0; nt < 4; ++nt) {
                            const int cb = 32 * nw + 8 * nt + 2 * (lane & 3);
                            const int h = nt >> 1;
                            const int nl = 16 * nw + 8 * (nt & 1) + 2 * (lane & 3);
                            uint32_t two = 0;
                            #pragma unroll
                            for (int i = 0; i < 2; ++i) {
                                float y = acc[mi][nt][2 * hr + i];
                                if constexpr (FP8) y = __fmul_rn(__fmul_rn(y, a_s), wsc[slot * BN + cb + i]);
                                two |= (uint32_t)bf16_bits_rn(y) << (16 * i);
                            }
                            const long col = (long)h * p.inter + n0 + nl;
                            *reinterpret_cast<uint32_t*>(out + (long)flat * p.out_stride + col) = two;
                        }
                    } else {
                        #pragma unroll
                        for (int nt = 0; nt < 4; ++nt) {
                            const int cb = 32 * nw + 8 * nt + 2 * (lane & 3);
                            uint32_t two = 0;
                            #pragma unroll
                            for (int i = 0; i < 2; ++i) {
                                float y = acc[mi][nt][2 * hr + i];
                                if constexpr (FP8) y = __fmul_rn(__fmul_rn(y, a_s), wsc[slot * BN + cb + i]);
                                if (!DENSE && p.mul_weight) y = __fmul_rn(y, rw);
                                two |= (uint32_t)bf16_bits_rn(y) << (16 * i);
                            }
                            const long col = n0 + cb;
                            *reinterpret_cast<uint32_t*>(out + (long)flat * p.out_stride + col) = two;
                        }
                    }
                }
            }
            ++item_idx;
        }
    }
}

// out[t, :] = bf16( sum_{j < top_k} f32(routed[t * top_k + j, :]) ), fixed order.
__global__ void token_sum_kernel(const uint16_t* __restrict__ routed, uint16_t* __restrict__ out,
                                 long tokens, int top_k, long width) {
    const long vec = (long)blockIdx.x * blockDim.x + threadIdx.x;   // one 8-column vector
    const long vecs_per_row = width / 8;
    if (vec >= tokens * vecs_per_row) return;
    const long t = vec / vecs_per_row;
    const long c = (vec - t * vecs_per_row) * 8;
    float acc[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.f;
    for (int j = 0; j < top_k; ++j) {
        const uint4 v = *reinterpret_cast<const uint4*>(routed + (t * top_k + j) * width + c);
        const uint32_t w[4] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc[2 * i] += bf16_bits_to_f32(w[i] & 0xFFFFu);
            acc[2 * i + 1] += bf16_bits_to_f32(w[i] >> 16);
        }
    }
    uint4 o;
    o.x = bf16_bits_rn(acc[0]) | ((uint32_t)bf16_bits_rn(acc[1]) << 16);
    o.y = bf16_bits_rn(acc[2]) | ((uint32_t)bf16_bits_rn(acc[3]) << 16);
    o.z = bf16_bits_rn(acc[4]) | ((uint32_t)bf16_bits_rn(acc[5]) << 16);
    o.w = bf16_bits_rn(acc[6]) | ((uint32_t)bf16_bits_rn(acc[7]) << 16);
    *reinterpret_cast<uint4*>(out + t * width + c) = o;
}

int max_dynamic_smem_bytes(int device) {
    int v = 0;
    TESSERA_HOST_CUDA(cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    return v;
}

template <bool FP8, int MODE, bool DENSE, bool SPLIT, int RL, bool TWO>
void launch_pair(const Params& p, int grid, cudaStream_t stream) {
    const int smem = smem_bytes(MODE, p.slot_words);
    static int attributed = 0;     // the largest dynamic size this instantiation was granted
    if (smem > attributed) {
        TESSERA_HOST_CUDA(cudaFuncSetAttribute(routed_fused_kernel<FP8, MODE, DENSE, SPLIT, RL, TWO>,
                                                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        attributed = smem;
    }
    routed_fused_kernel<FP8, MODE, DENSE, SPLIT, RL, TWO><<<grid, THREADS, smem, stream>>>(p);
    TESSERA_HOST_CUDA(cudaGetLastError());
}

// The launch's run pair, from its ``tile_words`` (``pair_of``), picks the
// kernel instantiation.  A pair this launch does not decode
// (``launch_decodes``) is refused here, before any launch; the Python owner
// refuses it first (``fused_routed_window_supported``: the device's shared
// memory; ``run_pair``: adjacency).
template <bool FP8, int MODE, bool DENSE = false, bool SPLIT = false>
void launch(const Params& p, int grid, cudaStream_t stream) {
    const PairKey k = pair_of(p.tile_words, p.K);
    switch (k.two ? RATE_MAX + k.r_lo : k.r_lo) {
#define TESSERA_ROUTED_FUSED_PAIR(R, T)                                                        \
        case (T ? RATE_MAX : 0) + R:                                                           \
            if constexpr (launch_decodes(MODE, R, T)) {                                        \
                launch_pair<FP8, MODE, DENSE, SPLIT, R, T>(p, grid, stream);                   \
                return;                                                                        \
            }                                                                                  \
            break;
        TESSERA_ROUTED_FUSED_PAIR(1, false) TESSERA_ROUTED_FUSED_PAIR(2, false)
        TESSERA_ROUTED_FUSED_PAIR(3, false) TESSERA_ROUTED_FUSED_PAIR(4, false)
        TESSERA_ROUTED_FUSED_PAIR(5, false) TESSERA_ROUTED_FUSED_PAIR(6, false)
        TESSERA_ROUTED_FUSED_PAIR(7, false) TESSERA_ROUTED_FUSED_PAIR(8, false)
        TESSERA_ROUTED_FUSED_PAIR(1, true) TESSERA_ROUTED_FUSED_PAIR(2, true)
        TESSERA_ROUTED_FUSED_PAIR(3, true) TESSERA_ROUTED_FUSED_PAIR(4, true)
        TESSERA_ROUTED_FUSED_PAIR(5, true) TESSERA_ROUTED_FUSED_PAIR(6, true)
        TESSERA_ROUTED_FUSED_PAIR(7, true)
#undef TESSERA_ROUTED_FUSED_PAIR
        default: break;
    }
    TESSERA_HOST_REFUSE_PAIR(MODE, k.r_lo, k.two, p.tile_words, p.K);
}

// DENSE && SPLIT: out[m, n] = epilogue( sum_{s < S} partial[s, m, n] ), the sum
// in fixed split order in fp32, then the same operation order as the unsplit
// epilogue: ``(acc * a_scale[m]) * w_scale[n]`` for the E4M3 family, the bare
// accumulator for the folded value family, one bf16 rounding.
template <bool FP8>
__global__ void dense_reduce_kernel(const float* __restrict__ partial, const float* __restrict__ a_scale,
                                    const float* __restrict__ wscale, uint16_t* __restrict__ out,
                                    long out_stride, int S, long M, long N) {
    const long quad = (long)blockIdx.x * blockDim.x + threadIdx.x;   // four consecutive columns
    const long quads_per_row = N / 4;
    if (quad >= M * quads_per_row) return;
    const long m = quad / quads_per_row;
    const long n = (quad - m * quads_per_row) * 4;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int s = 0; s < S; ++s) {
        const float4 v = *reinterpret_cast<const float4*>(partial + ((long)s * M + m) * N + n);
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    float y[4] = {acc.x, acc.y, acc.z, acc.w};
    if constexpr (FP8) {
        const float a_s = a_scale[m];
        #pragma unroll
        for (int i = 0; i < 4; ++i) y[i] = __fmul_rn(__fmul_rn(y[i], a_s), wscale[n + i]);
    }
    uint2 o;
    o.x = bf16_bits_rn(y[0]) | ((uint32_t)bf16_bits_rn(y[1]) << 16);
    o.y = bf16_bits_rn(y[2]) | ((uint32_t)bf16_bits_rn(y[3]) << 16);
    *reinterpret_cast<uint2*>(out + m * out_stride + n) = o;
}

}  // namespace

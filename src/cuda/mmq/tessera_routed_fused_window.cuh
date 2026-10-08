// VENDORED from Tessera (https://github.com/RobTand/tessera), src/tessera/serving/csrc/routed_fused_window.cu
// at upstream 37742e0f (2026-10-08), file sha256 4e93959a2a265dcd560e827fe65bbfe703b25b0e7769aef221bd426b882c802a.
// Made with Tessera by Robert Tand - https://github.com/RobTand/tessera
// License: MIT + Tessera Attribution Addendum 1.0 (LicenseRef-Tessera-Attribution-1.0); see src/cuda/mmq/VENDOR-TESSERA.md.
//
// pulsar edits (host glue only -- every device function is byte-for-byte upstream):
//   * the torch / c10 / ATen #includes (upstream lines 68-72) are dropped;
//   * the c10 CUDA checks in max_dynamic_smem_bytes / launch_variant call the includer's TESSERA_HOST_CUDA(expr);
//   * the three piece-major TORCH_CHECKs in launch call TESSERA_HOST_CHECK(cond, msg), and its run-pair refusal
//     calls TESSERA_HOST_REFUSE_PAIR(mode, r_lo, two, tile_words, K);
//   * check_words / check_slot / i32_ptr / f32_ptr / TABLE_DTYPE / check_run_tables (torch::Tensor helpers of the
//     host entries, upstream 2063-2080 and 2110-2131), the E2M1 family (2132-3048, built only at
//     TESSERA_ROUTED_FUSED_FP4=1) and the torch host entries + PYBIND module (3051-end) are dropped; pulsar's
//     launcher (pulsar_tessera.cu) replaces the entries.  dense_reduce_kernel (upstream 2081-2109) is kept verbatim.
// The includer defines TESSERA_ROUTED_FUSED_FP8 and the three hooks before including this file, and compiles the
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
// stream (``rate`` bits per row per column: 1..8 on the routed launches, up
// to RATE_MAX on the dense one -- the run table's
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
// split ``S`` ways (``k_split``, at most ``K / 64`` so every split item keeps
// two K chunks; see ``dense_forward``); each split accumulates its chunk range into
// an fp32 workspace ``[S, M, N]`` and ``dense_reduce_kernel`` sums the ``S``
// partials in fixed order and applies the epilogue -- deterministic, and the
// same fp32 operation order as the unsplit epilogue once the sum is formed.
// On the E4M3 libraries (``MULTI``, tessera#750 WP2) one launch takes all of a
// merged Linear's roles -- their 128-row blocks are one item list, each role's
// planes its own tensors (``DenseRoles``) -- and the split is reduced in the
// kernel: the last split of a tile to arrive (a per-tile arrival count) sums
// the partials in the same fixed order and applies the same epilogue, so the
// output is bitwise the two-launch one.  A split reduced in the kernel keeps
// at least STAGES + 1 K chunks per item, which is what keeps an item's
// descriptor and row-scale slot live until its fixup is done with it.
// The dense identity is ``tessera::fused_window_dense`` (``serving.native_
// window``), decoders ``native_fused_window_dense`` / ``..._folded``.
//
// The Python owner is ``tessera.routed_fused``; the contract publishes this
// file as two ``native_extensions`` entries (one per family, see ``ext``).

// (upstream lines 68-72: the torch / c10 / ATen includes -- dropped, see the header above)
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>
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
// The E4M3 family's tensor-core instruction.  0 (``tessera_routed_fused_e4m3``)
// widens the E4M3 table entry and activation to f16 -- both exactly -- and
// runs ``mma.sync.m16n8k16.f32.f16.f16.f32``.  1
// (``tessera_routed_fused_mma_e4m3``) keeps both operands E4M3 and runs
// ``mma.sync.m16n8k32.f32.e4m3.e4m3.f32``: the same exact products at twice
// the instruction's K, one fp32 truncation per 32 products instead of per 16,
// and byte-wide tables and tiles (see ``E4M3 MMA LAYOUT`` below).  Its own
// library and decoders, because its accumulation order differs.
#ifndef TESSERA_ROUTED_FUSED_MMA8
#define TESSERA_ROUTED_FUSED_MMA8 0
#endif
constexpr bool FAMILY_MMA8 = TESSERA_ROUTED_FUSED_MMA8 != 0;
// Experimental compile gate only; the same wire/forward ABI and original
// specializations remain available in this binary. No serving promotion.
#ifndef TESSERA_ROUTED_FUSED_PAIRED_K32
#define TESSERA_ROUTED_FUSED_PAIRED_K32 0
#endif
constexpr bool PAIRED_K32_BUILD = TESSERA_ROUTED_FUSED_PAIRED_K32 != 0 && FAMILY_MMA8;
static_assert(!FAMILY_MMA8 || FAMILY_FP8, "the E4M3 instruction serves the E4M3 family only");
// The E2M1 family (``tessera_routed_fused_e2m1``, -DTESSERA_ROUTED_FUSED_FP4=1
// with FP8=0): the E2M1x2 window wire over its LUT16 group-scale plane, on the
// block-scaled FP4 instruction (see THE E2M1 FAMILY below).  Its kernel, its
// host entries and its module attributes are compiled only into its own
// library, which is built for the architecture-specific target
// (``backend.offload_flags(token, arch_specific=True)``: ptxas refuses the
// instruction for the generic one); the other three libraries compile none of
// it, and none of theirs is compiled into it.
#ifndef TESSERA_ROUTED_FUSED_FP4
#define TESSERA_ROUTED_FUSED_FP4 0
#endif
constexpr bool FAMILY_FP4 = TESSERA_ROUTED_FUSED_FP4 != 0;
static_assert(!FAMILY_FP4 || (!FAMILY_FP8 && !FAMILY_MMA8), "the E2M1 family is its own library");

constexpr int THREADS = 512;
constexpr int PRODUCER_THREADS = 256;
constexpr int BM = 64;                              // routes per superblock
constexpr int BN = 128;                             // B columns per item (two halves)
constexpr int HALF = 64;
constexpr int BK = 32;                              // k columns per chunk
// The dense identity's row quantum (N-tail): a role's last BN block may be
// partial.  The epilogue stores SEGW <= 4 columns at a time and the split
// reduce four, so a role's rows are a multiple of 4 and each store is wholly
// inside the role or wholly past it.
constexpr int DENSE_ROW_QUANTUM = 4;
// Column rates the decode reads (bits per code): a column's 512-row chunk is
// ``16 * rate`` int32 words, a 64-row half of it ``2 * rate`` words.  Rates 4
// and 8 put a lane's eight rows on word boundaries; every other rate re-aligns
// the lane's window once per chunk (``decode_rows``).  The run table's rates
// are the grammar's -- one rate, or the two ADJACENT rates bracketing the root
// (``grammar.rate_set``); a non-adjacent pair is refused by name
// (``routed_fused.run_pair``) and traps here -- and every rate up to RATE_MAX
// is read on the dense launch, up to ROUTED_RATE_MAX on the routed ones.
// Each (low rate, one or two runs) pair is its own KERNEL instantiation
// (``routed_fused_kernel<..., RL, TWO>``), chosen on the host from the
// launch's ``tile_words`` (``pair_of``): one launch carries one pair, so each
// pair gets its own register allocation and scheduling instead of sharing
// one kernel's with every other pair (``launch_decodes``).
constexpr int RATE_MIN = 1;
// The largest rate this library decodes, on its dense launch.  A code's
// window is WINDOW_BITS = 14 bits, so the value family's BF16 grid reads rates
// 1..14 (tessera#750 item 4: rates 15 and 16 widen the window to 15 and 16
// bits, whose 64 KB and 128 KB tables leave the one-table block one word
// stage and none).  The E4M3 grids' codes are 8 bits, and so are the E2M1
// family's (a code is two E2M1 nibbles).
constexpr int RATE_MAX = (FAMILY_FP8 || FAMILY_FP4) ? 8 : 14;
// The routed-expert launches (gate/up and down) read 1..8 in every family.
// The two-table gate/up launch has no room for a slot above rate 8 even at two
// word stages (101,840 B at rate 9 against sm_121's 101,376 B), and the down
// launch reads the same run tables as the stack's gate/up.
constexpr int ROUTED_RATE_MAX = 8;
constexpr int BDESC_INTS = 12;                      // per-32-column descriptor (see ``col_map``)
constexpr int TILE_ROWS = 512;
// The 64-row pieces inside one 512-row tile (the piece-major layout's plan:
// a column's chunk is PIECES_PER_TILE pieces of 2*rate words each).
constexpr int PIECES_PER_TILE = TILE_ROWS / 64;
constexpr int WINDOW_BITS = 14;
constexpr int TABLE_ENTRIES = 1 << WINDOW_BITS;
static_assert(RATE_MAX <= WINDOW_BITS && ROUTED_RATE_MAX <= RATE_MAX, "a code fits its window");
constexpr int STAGES = 2;
// The word stages a launch's chunk loop cycles through: three (words issued
// two chunks ahead of their decode) wherever the pair's slot fits the block at
// three, and WORD_STAGES_MIN (one chunk ahead) where it does not -- only the
// 16-bit libraries' two-table gate/up launch at slot 16, the pairs that read a
// rate-7 or rate-8 run (``word_stages``).
constexpr int WORD_STAGES = 3;
constexpr int WORD_STAGES_MIN = 2;
// The activation prefetch distance, in chunks (0: none).  Each chunk's A row
// is loaded into registers one chunk ahead (``load_a``), and the chunk loop's
// last register move waits for it.  NCU on the routed launch (M = 1, 512 and
// 2048) put that wait on the producers' critical path: the warps that stage A
// rows stalled on it, and every other producer waited for them at the
// producers' barrier.  A ``prefetch.global.L1`` of the row A_PREFETCH chunks
// ahead brings the line in while earlier chunks decode.
//
// Measured on the E4M3 instruction's routed launches only (recorded prefill
// routing, drift-symmetric, output bitwise equal; pf4one-ab-20260930T*): at
// distance 4 the one-run R1024 launch ran at 0.847 (M 512) and 0.861
// (M 2048) of no prefetch, where removing the A load altogether bounds it at
// 0.82-0.83; on the two-run R1088 launch the same prefetch cost 2.7-4.2%,
// and gated off there it measured 1.000/0.998.  So the kernel prefetches on
// the launches that measurement covers -- the E4M3 instruction's routed
// one-run pairs (``PREFETCH_A``) -- and nowhere else.
#ifndef TESSERA_ROUTED_FUSED_A_PREFETCH
#define TESSERA_ROUTED_FUSED_A_PREFETCH 4
#endif
constexpr int A_PREFETCH = TESSERA_ROUTED_FUSED_A_PREFETCH;
// Experimental mode-0 one-run R4 consumer schedule (tessera#739). Both
// independent B column groups load before the first group's MMAs. Register
// lifetime changes, but accumulator K order and shared ownership do not.
#ifndef TESSERA_ROUTED_FUSED_MMA8_GATE_UP_B_PREFETCH
#define TESSERA_ROUTED_FUSED_MMA8_GATE_UP_B_PREFETCH 0
#endif
static_assert(TESSERA_ROUTED_FUSED_MMA8_GATE_UP_B_PREFETCH == 0 ||
              TESSERA_ROUTED_FUSED_MMA8_GATE_UP_B_PREFETCH == 1, "B schedule is 0 or 1");
constexpr bool MMA8_GATE_UP_B_PREFETCH =
    FAMILY_MMA8 && TESSERA_ROUTED_FUSED_MMA8_GATE_UP_B_PREFETCH;
static_assert(A_PREFETCH == 0 || A_PREFETCH >= 2, "distance 1 is the load itself");
// The activation ring on the E4M3 instruction (tessera#739), default off.
// Instead of loading each chunk's A row into registers one chunk ahead
// (``load_a``), whose wait is the chunk loop's last register move, every
// A-staging producer copies its 16 raw E4M3 bytes (``cp_async16``) two chunks
// ahead into a per-word-stage slot of its own, in the word stages' commit
// group, so the chunk's ``cp_async_wait`` covers it; ``store_a`` then reads the
// slot back from shared memory and writes the same fragment-order bytes as
// before.  Only the thread that copies a slot ever reads it, so no barrier is
// added.  The bytes, the A tile and every MMA are unchanged: the output is
// bitwise the register path's.  At 1 the ring serves the routed two-run
// launches only.  Measured against the no-load ceiling (the A load replaced by
// a register value; opus-739-20261004T182357Z, T8R release stacks, every cell
// bitwise): the two-run routed R1088/R832 launches run at 0.90-0.94 of master
// for M >= 512 (ceiling 0.87-0.94); the one-run R1024 launch's ceiling is
// 0.97-0.99 behind ``prefetch_a``, which the ring did not beat (at most
// +3.1%); and the dense and shared launches have no load wait to hide
// (ceiling ~1.0)
// and lost up to 5% to the ring's extra instructions
// (docs/measurements/2026-10-04-mma8-activation-ring.md).  The ring is
// WORD_STAGES * bmt * BK bytes after the A tiles (``a_region_bytes``),
// allocated in every launch's layout when the flag is on.
#ifndef TESSERA_ROUTED_FUSED_MMA8_A_RING
#define TESSERA_ROUTED_FUSED_MMA8_A_RING 0
#endif
static_assert(TESSERA_ROUTED_FUSED_MMA8_A_RING >= 0 && TESSERA_ROUTED_FUSED_MMA8_A_RING <= 1,
              "the activation ring is 0 or 1");
constexpr int MMA8_A_RING = FAMILY_MMA8 ? TESSERA_ROUTED_FUSED_MMA8_A_RING : 0;
// Folded BF16 qualification arm (tessera#874), never enabled by default.
// Reuses load_a addressing; no additional shared-memory allocation.
#ifndef TESSERA_ROUTED_FUSED_VALUE_A_PREFETCH
#define TESSERA_ROUTED_FUSED_VALUE_A_PREFETCH 0
#endif
constexpr int VALUE_A_PREFETCH = TESSERA_ROUTED_FUSED_VALUE_A_PREFETCH;
static_assert(VALUE_A_PREFETCH == 0 || VALUE_A_PREFETCH >= 2,
              "value prefetch distance 1 is the load itself");

// One table entry and one A/B tile element: 16-bit, or one E4M3 byte on the
// E4M3 instruction.
constexpr int ELEM_BYTES = FAMILY_MMA8 ? 1 : 2;
constexpr int TABLE_BYTES = TABLE_ENTRIES * ELEM_BYTES;         // 32768 (16384), one table
constexpr int B_STAGE_BYTES = BK * BN * ELEM_BYTES;             // 8192 (4096)
constexpr int A_STAGE_BYTES = BM * BK * ELEM_BYTES;             // 4096 (2048)
using TabT = std::conditional_t<FAMILY_MMA8, uint8_t, uint16_t>;
constexpr int WSCALE_FLOATS = 2 * BN;                           // two item slots
constexpr int DESC_INTS = 2 * 8;
// The shared-memory layout.  The word stages come LAST and are sized at
// launch by the stack's rates: ``Params::slot_words`` int32 words per (half,
// column) slot -- ``slot_words_for_rate`` of the larger rate (2 * rate, plus
// the two words the odd-rate copies start early by), rounded to a multiple of
// 4 so the 16-byte copies stay aligned.  A block on sm_121 may opt in to
// 101,376 B of dynamic shared memory; the two-table gate/up launch needs
// 91,600 + 256 * slot_words per word stage, so at three stages it fits slots
// up to 12 words (rates <= 6) and not the 16-word slot of rates 7 and 8
// (103,888 B); there it runs two stages (99,792 B, ``word_stages``).  The
// one-table down/dense launch (MODE 2) fits every rate at three.  The host
// entries check the launch against the device's own limit.
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
// THE STAGED STREAM HISTORY (tessera#750).  A half's decode needs the 32
// stream bits before its first word (``load_prev``).  That word sits outside
// the half's copied words, so the chunk loop loaded it from global memory one
// chunk ahead, into a register the loop's last move waited on.  NCU on the
// R1024 gate/up launch at M = 512 (E4M3 instruction): that register move
// (``MOV R66, R72``, the half-1 word) was the kernel's hottest instruction, 7.5%
// of all warp samples, all long scoreboard, and the producers' barrier behind
// it another 10.2%: each chunk waited one global latency, whatever the number
// of producer warps (sixteen measured no faster).  On the E4M3 instruction the
// word therefore rides the word stages' own copies instead: ``issue_words``
// copies it (4 bytes, ``cp_async4``) into a per-stage slot two chunks ahead,
// the chunk's ``cp_async_wait`` and producers' barrier cover it, and the
// decode reads it from shared memory.  It costs WORD_STAGES * 2 * BK int32
// (768 B) of shared memory, which the 16-bit libraries' two-table gate/up
// layout cannot spare at rates 5 and 6 (560 B of headroom), so the value
// family and the 16-bit E4M3 library keep the register path.
constexpr bool PREV_STAGED = FAMILY_MMA8;
constexpr int PREV_STAGE_INTS = 2 * BK;             // one word per (half, column)
constexpr int PREV_REGION_BYTES = PREV_STAGED ? WORD_STAGES * PREV_STAGE_INTS * 4 : 0;
// The wide superblock (tessera#741): 128 routes per item instead of BM = 64,
// so one decoded B tile feeds twice the rows.  Its A region is twice the
// size (``a_region_bytes``), which fits the one-table down/dense launch at
// every rate in both E4M3 libraries and the two-table gate/up launch at every
// rate on the E4M3 instruction, whose 8-bit tiles leave the room; the 16-bit
// gate/up layout fits it at none (``has_width``).  The consumers skip every
// 16-row MMA block that lies wholly past the item's routes (``mb``), at either
// width, so a wide item whose routes fit 64 rows issues the MMAs a 64-route
// item would.  Each output row keeps its own K order and its own epilogue, so
// the output is bitwise the 64-route launch's; the host picks the width per
// launch (``routed_fused.superblock_rows``).  The value family stages four
// threads per A row, so it keeps BM.
constexpr int BM_WIDE = 128;
// The A region: STAGES tiles of ``bmt`` rows, then (MMA8_A_RING) the
// activation ring's WORD_STAGES raw tiles.
constexpr int A_RING_STAGES = MMA8_A_RING ? WORD_STAGES : 0;
__host__ __device__ constexpr int a_region_bytes(int bmt) { return (STAGES + A_RING_STAGES) * bmt * BK * ELEM_BYTES; }
static_assert(a_region_bytes(BM) == (STAGES + A_RING_STAGES) * A_STAGE_BYTES, "the 64-route layout is the published one");
template <int MODE, int BMT = BM, bool PAIRED = false> struct Layout {
    static constexpr int MICROS = PAIRED ? 2 : 1;
    static constexpr int TABLES = (MODE == 2) ? 1 : 2;
    static constexpr int PROJ = (MODE == 2) ? 1 : 2;             // projections per item
    static constexpr int DRING_STAGE = PROJ * BDESC_INTS;        // int32 per ring slot
    static constexpr int OFF_TABLES = 0;
    static constexpr int OFF_B = OFF_TABLES + TABLES * TABLE_BYTES;
    static constexpr int OFF_A = OFF_B + STAGES * MICROS * B_STAGE_BYTES;
    static constexpr int OFF_WSCALE = OFF_A + MICROS * a_region_bytes(BMT);
    static constexpr int OFF_DESC = OFF_WSCALE + WSCALE_FLOATS * 4;
    static constexpr int OFF_CLAIM = OFF_DESC + DESC_INTS * 4;
    static constexpr int OFF_DRING = OFF_CLAIM + 16;
    // The staged stream history (PREV_STAGED; empty on the 16-bit libraries).
    static constexpr int OFF_PREV = OFF_DRING + DRING_STAGES * DRING_STAGE * 4;
    // 91,600 (two tables) / 58,640 (one); 47,312 / 30,736 on the E4M3 instruction
    static constexpr int OFF_W = OFF_PREV + (PAIRED ? STAGES * MICROS * PREV_STAGE_INTS * 4 : PREV_REGION_BYTES);
    static_assert(OFF_W % 16 == 0, "the word stages take 16-byte copies");
    static_assert(OFF_DRING % 16 == 0, "the descriptor ring takes 16-byte copies");
};
constexpr int SLOT_WORDS_MAX = 2 * RATE_MAX;                    // the RATE_MAX slot: 28 (value), 16 (E4M3)
__host__ __device__ constexpr int w_stage_ints(int slot_words) { return 2 * BK * slot_words; }
__host__ __device__ constexpr int smem_bytes_ws(int mode, int slot_words, int word_stages) {
    return (mode == 2 ? Layout<2>::OFF_W : Layout<0>::OFF_W) + word_stages * w_stage_ints(slot_words) * 4;
}
// The per-block dynamic shared memory sm_121 (GB10, the contract's target)
// lets a kernel opt in to.  It bounds which pairs are INSTANTIATED, and at how
// many word stages; every launch is checked against the live device's own
// limit (``check_slot``), and ``routed_fused.SM121_MAX_DYNAMIC_SMEM`` is the
// same figure.
constexpr int SM121_SMEM_OPTIN = 101376;
// The word stages a launch of ``mode`` at ``slot_words``-word slots takes:
// WORD_STAGES where they fit the target's block, else WORD_STAGES_MIN.  A
// pure function of the two, so the host sizes the block the kernel lays out
// (``routed_fused.word_stages``).
__host__ __device__ constexpr int word_stages(int mode, int slot_words) {
    return smem_bytes_ws(mode, slot_words, WORD_STAGES) <= SM121_SMEM_OPTIN ? WORD_STAGES : WORD_STAGES_MIN;
}
__host__ __device__ constexpr int smem_bytes(int mode, int slot_words) {
    return smem_bytes_ws(mode, slot_words, word_stages(mode, slot_words));
}
// The same at ``bmt``-route superblocks: the published figure at BM, the
// larger A region at BM_WIDE.
__host__ __device__ constexpr int smem_bytes_at(int mode, int slot_words, int bmt) {
    return smem_bytes(mode, slot_words) + a_region_bytes(bmt) - a_region_bytes(BM);
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
// The dynamic shared memory a launch at ``slot_words``-word slots and
// ``bmt``-route superblocks takes.
__host__ __device__ constexpr int launch_smem_bytes(int mode, int slot_words, int bmt = BM, bool paired = false) {
    if (paired) {
        // Paired is admitted only at the R4/BMT128 geometry. Keep the reserved
        // descriptor ring; only word/history and decoded stage multiplicity move.
        return (mode == 2 ? Layout<2, BM_WIDE, true>::OFF_W : Layout<0, BM_WIDE, true>::OFF_W)
             + STAGES * 2 * w_stage_ints(slot_words) * 4;
    }
    return smem_bytes_at(mode, slot_words, bmt);
}
// Runtime shape rule shared by the entry's live-device guard and dispatch.
// Three paired quanta per item are REQUIRED: wscale is read in the epilogue
// after the final EMPTY, so two pairs do not protect its next item-slot reuse.
__host__ __device__ constexpr bool paired_k32_scope(bool fp8, bool mma8, int mode,
        bool dense, bool split, int r_lo, bool two, int bmt, int K, long rows, int slot_words = 8) {
    return PAIRED_K32_BUILD && fp8 && mma8 && (mode == 0 || mode == 2)
        && !dense && !split && r_lo == 4 && !two && bmt == BM_WIDE
        && K >= 6 * BK && K % (2 * BK) == 0 && rows >= 512 && slot_words == pair_slot_words(4, false);
}
// Whether the launch of ``mode`` exists at ``bmt``-route superblocks in a
// library of the family (``fp8``) and instruction (``mma8``): BM everywhere;
// BM_WIDE in the E4M3 family for the one-table launch, and for gate/up too on
// the E4M3 instruction.  Where it exists it fits every rate the launch
// decodes (the kernel asserts it per pair).
__host__ __device__ constexpr bool has_width(bool fp8, bool mma8, int mode, int bmt) {
    return bmt == BM || (bmt == BM_WIDE && fp8 && (mode == 2 || mma8));
}
// Whether the launch of ``mode`` decodes the pair (``r_lo``; ``two``: a second
// run at ``r_lo + 1``): rates in 1..ROUTED_RATE_MAX on the routed launches and
// 1..RATE_MAX on the dense one (``dense``), and the pair's slot fits the
// target's block at its word stages.  Every launch reaches every rate and pair
// of its range on sm_121: the two-table gate/up launch of the 16-bit libraries
// at two word stages for the pairs that read a rate-7 or rate-8 run, three
// elsewhere; the value family's dense launch at three up to rate 14 (80,144 B).
// Only these pairs are instantiated; the host refuses any other before a
// launch (``launch``), as ``routed_fused.fused_routed_window_supported`` and
// ``fused_dense_window_supported`` do first.
__host__ __device__ constexpr bool launch_decodes(int mode, int r_lo, bool two, bool dense = false) {
    return r_lo >= RATE_MIN && r_lo + (two ? 1 : 0) <= (dense ? RATE_MAX : ROUTED_RATE_MAX)
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
    for (int x = RATE_MIN; x <= ROUTED_RATE_MAX; ++x)
        if (launch_decodes(0, x, false)) r = x;
    return r;
}

constexpr int BAR_FULL0 = 1;
constexpr int BAR_EMPTY0 = 3;
constexpr int BAR_PROD = 5;
constexpr int BAR_CONS = 6;       // MULTI && SPLIT: the consumers' split fixup

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
// The 4-byte copy is the staged stream history's word (PREV_STAGED).
__device__ __forceinline__ void cp_async4(void* smem, const void* gmem) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void prefetch_l1(const void* gmem) {
    asm volatile("prefetch.global.L1 [%0];" :: "l"(gmem));
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
// E4M3 MMA LAYOUT (FAMILY_MMA8).  ``m16n8k32.e4m3`` wants four consecutive k
// of one row (A) or one column (B) per register, and the decoder produces
// eight consecutive n of one k.  The B tile stays [k][n], one byte per weight
// (128 B per k row), and is read with the 16-bit TRANSPOSING ldmatrix: a
// thread then holds, for k rows (2q, 2q + 1) of each 8-row matrix, the byte
// pairs of n (2g, 2g + 1).  Two byte permutes split each pair of registers
// into an even-n and an odd-n fragment whose four k are {2q, 2q + 1, 2q + 8,
// 2q + 9} (+16 for the second register), so one 16-column group feeds two
// MMAs: the even one serves columns 16G + 2g, the odd one 16G + 2g + 1, and
// thread (g, q) ends up owning the four consecutive columns 16G + 4q .. + 3.
// The A tile is staged in the SAME k order -- logical k' = 4q + i is physical
// k = 2q + (i & 1) + 8 (i >> 1) within each 16-column half -- by four byte
// permutes per staging thread, so the MMA's K sum pairs every product exactly
// once.  B's 16-byte units are XOR-swizzled by ``b8swz`` of the k row: eight
// consecutive k rows land on eight distinct units (the transposing ldmatrix
// reads conflict-free) and rows (2p, 2p + 1) differ in unit bits 0 and 2, so a
// warp's 8-byte decode stores for four consecutive columns take the two
// wavefronts 256 bytes need, in both the down and the gate/up chunk maps.
// A rows are 32 B; their two 16-byte units swap every four rows.
__device__ __forceinline__ int b8swz(int k) {
    return ((k >> 1) & 3) ^ ((k & 1) * 5);
}
__device__ __forceinline__ int b8off(int chunk, int k) {     // chunk: eight n
    return ((((chunk >> 1) ^ b8swz(k)) & 7) << 4) | ((chunk & 1) << 3);
}
__device__ __forceinline__ void mma16832_e4m3(float (&d)[4], const uint32_t (&a)[4],
                                              const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// Four table bytes (each in a register's low byte) as one word, first lowest.
__device__ __forceinline__ uint32_t pack4(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    return __byte_perm(__byte_perm(a, b, 0x0040), __byte_perm(c, d, 0x0040), 0x5410);
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
// ``u`` into it.  The lane re-aligns an NZ-word window on ``u`` once (the 32
// stream bits before its first row, then its eight fields' 8 * R bits: three
// words up to rate 8, four at rates 9..12, five at 13 and 14), after which
// every row's field sits at a compile-time position; at rates 4, 8 and 12 ``u``
// is 0 and the window is the slot's words themselves -- the rate-4 path is
// exactly the original kernel's constant shifts.  ``prev`` is the word before
// the half's first word (the previous 64 rows, the previous tile's last word of
// the column, the cut's start state, or zero).
//
// Two forms of one window.  Rates 1..8 (NZ <= 3) keep the three named words
// ``Z0 | Z1 | Z2`` and the statements of the kernel that contract v49 attests,
// so every rate-1..8 instantiation compiles to that kernel's SASS: an array of
// NZ words moves ptxas's schedule in 21 of the E4M3 libraries' 244 kernels
// (tessera#750 item 4; the receipt is in
// docs/measurements/2026-09-30-t16-dense-rates-9-14.md).  Rates 9..14 read the
// NZ-word array ``Zw``.  One form can replace both once a same-session timing
// A/B shows the array form costs nothing at rates 1..8.
template <bool FP8, int R>
__device__ __forceinline__ void decode_rows(const int32_t* Wc, uint32_t prev, int j,
                                            const TabT* T, const float* ws,
                                            uint32_t (&packed)[4]) {
    constexpr bool ALIGNED = (8 * R) % 32 == 0;
    constexpr int NZ = 1 + (8 * R + 31) / 32;         // the window's words: 2 (R <= 4) .. 5 (R = 14)
    const int bits0 = 8 * j * R;
    const int b = bits0 >> 5;
    const uint32_t wm1 = (b > 0) ? (uint32_t)Wc[b - 1] : prev;
    const uint32_t w0 = (uint32_t)Wc[b];
    constexpr bool WIDE = 8 * R > 32;                 // rows reach past 32 bits: Z2 is read
    [[maybe_unused]] uint32_t Z0, Z1, Z2 = 0;         // rates 1..8
    [[maybe_unused]] uint32_t Zw[NZ > 3 ? NZ : 1];    // rates 9..14
    if constexpr (NZ <= 3) {
        if constexpr (ALIGNED) {
            Z0 = wm1;
            Z1 = w0;
            if constexpr (WIDE) Z2 = (uint32_t)Wc[b + 1];
        } else {
            const int u = bits0 & 31;                 // 1..31 here, 0 only for j = 0
            // The lane's eight fields end 8 * R bits after bits0; the next word
            // is read only where a field reaches into it, so the last lane of a
            // half never reads past the half's 2 * R words.
            const uint32_t w1 = (u + 8 * R > 32) ? (uint32_t)Wc[b + 1] : 0u;
            // the 32 stream bits starting u into (hi:lo), MSB-first; u = 0 gives hi
            Z0 = __funnelshift_rc(w0, wm1, 32 - u);
            Z1 = __funnelshift_rc(w1, w0, 32 - u);
            if constexpr (WIDE) {
                const uint32_t w2 = (u + 8 * R > 64) ? (uint32_t)Wc[b + 2] : 0u;
                Z2 = __funnelshift_rc(w2, w1, 32 - u);
            }
        }
    } else if constexpr (ALIGNED) {
        Zw[0] = wm1;
        Zw[1] = w0;
        #pragma unroll
        for (int k = 2; k < NZ; ++k) Zw[k] = (uint32_t)Wc[b + k - 1];
    } else {
        const int u = bits0 & 31;
        // w[k]: slot word b + k - 1 (w[0] the word before), read only where a
        // field reaches into it, as above.
        uint32_t w[NZ + 1];
        w[0] = wm1;
        w[1] = w0;
        #pragma unroll
        for (int k = 2; k <= NZ; ++k) w[k] = (u + 8 * R > 32 * (k - 1)) ? (uint32_t)Wc[b + k - 1] : 0u;
        #pragma unroll
        for (int k = 0; k < NZ; ++k) Zw[k] = __funnelshift_rc(w[k + 1], w[k], 32 - u);
    }
    [[maybe_unused]] uint32_t v8[8];                  // FAMILY_MMA8: the eight table bytes
    #pragma unroll
    for (int r = 0; r < 8; r += 2) {
        uint32_t s[2];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int e = (r + i + 1) * R;            // the field ends here, in the window
            const int k1 = (e - 1) >> 5;              // the field lies in window words k1, k1 + 1
            const int shift = 32 * (k1 + 1) - e;
            if constexpr (NZ <= 3) {
                const uint32_t lo = k1 ? Z2 : Z1;
                const uint32_t hi = k1 ? Z1 : Z0;
                s[i] = __funnelshift_r(lo, hi, shift) & 0x3FFFu;
            } else {
                s[i] = __funnelshift_r(Zw[k1 + 1], Zw[k1], shift) & 0x3FFFu;
            }
        }
        uint32_t t0 = T[s[0]], t1 = T[s[1]];
        if constexpr (!FP8) {
            // FOLDED: one bf16 rounding of value * row_scale, before the dot
            t0 = bf16_bits_rn(__fmul_rn(bf16_bits_to_f32(t0), ws[r]));
            t1 = bf16_bits_rn(__fmul_rn(bf16_bits_to_f32(t1), ws[r + 1]));
        }
        if constexpr (FAMILY_MMA8) {
            v8[r] = t0;
            v8[r + 1] = t1;
        } else {
            packed[r >> 1] = t0 | (t1 << 16);
        }
    }
    if constexpr (FAMILY_MMA8) {
        // eight consecutive n of one k, one byte each: the B tile's 8-byte store
        packed[0] = pack4(v8[0], v8[1], v8[2], v8[3]);
        packed[1] = pack4(v8[4], v8[5], v8[6], v8[7]);
    }
}
// Both halves of a chunk (rates RA and RB, each compile-time) as ONE
// straight-line block, so the scheduler can issue the second half's word and
// table loads while the first half's are in flight.
template <bool FP8, int RA, int RB>
__device__ __forceinline__ void decode_two(const int32_t* const (&Wc)[2], const int32_t (&prev)[2], int j,
                                           const TabT* T0, const TabT* T1,
                                           const float* const (&ws)[2], uint32_t (&packed)[2][4]) {
    decode_rows<FP8, RA>(Wc[0], (uint32_t)prev[0], j, T0, ws[0], packed[0]);
    decode_rows<FP8, RB>(Wc[1], (uint32_t)prev[1], j, T1, ws[1], packed[1]);
}

// MULTI: a merged Linear's roles, one launch.  The roles share K, the run
// pair, ``tile_words`` and ``slot_words`` (a module is one rung), so their
// 128-row blocks are one list: role r owns blocks ``block0[r] ..`` of it, and
// every per-role plane is the role's own tensor.  Read with ``role_pick``
// (compile-time indices, so the table stays in parameter space).
constexpr int MAX_ROLES = 8;
struct DenseRoles {
    int n;                         // roles, 1 .. MAX_ROLES
    int block0[MAX_ROLES];         // the role's first 128-row block in the launch's list
    int rows[MAX_ROLES];           // the role's rows (its N)
    int col[MAX_ROLES];            // the role's first column of ``out`` (and of ``partial``)
    const int32_t* words[MAX_ROLES];
    const void* table[MAX_ROLES];
    const int32_t* init[MAX_ROLES];
    const int32_t* has_init[MAX_ROLES];
    const float* wscale[MAX_ROLES];
    const int32_t* runs[MAX_ROLES];
    const int32_t* bdesc[MAX_ROLES];
};

template <class T>
__device__ __forceinline__ T role_pick(const T (&a)[MAX_ROLES], int r) {
    T v = a[0];
    #pragma unroll
    for (int i = 1; i < MAX_ROLES; ++i)
        if (r == i) v = a[i];
    return v;
}

struct Params {
    const void* x;                 // [rows_x, K] bf16 (value) or e4m3 (fp8)
    const float* a_scale;          // [rows_x] fp32 (fp8) or nullptr
    const int32_t* words0;         // [E, words_stride] gate (mode 0/1) or down (mode 2)
    const int32_t* words1;         // [E, words_stride] up (mode 0/1) or nullptr
    const void* table0;            // [E, 16384] TabT (16-bit, or E4M3 bytes on FAMILY_MMA8)
    const void* table1;
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
    int piece_major;               // 1: words are [tile][piece][column] (tessera#739)
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
    // ---- appended (tessera#750 WP2): the E4M3 libraries' dense launch (MULTI)
    // takes a module's roles in one launch and reduces its K split in-kernel.
    // The fields above keep their offsets, so no other launch moves.
    int fixup;                     // MULTI && SPLIT: 1 = the last split of a tile reduces it
    int32_t* tile_sem;             // MULTI && SPLIT && fixup: [n_blocks * superblocks] arrivals
    DenseRoles roles;              // MULTI: the module's roles (N = the module's total rows)
};

// ``RL``, ``TWO``: the launch's run pair (``pair_of``) -- the low (or only)
// rate, and whether a second run at ``RL + 1`` exists.
// ``PM``: the resident words are the piece-major [tile][piece][column] order
// (tessera#739).  It is the bounded one-run rate-4 routed body only: an odd
// rate's 16-byte alignment depends on the piece index in a way the piece-major
// address (t64 * ncols + c) does not preserve, so the odd-rate copy path is
// deliberately not ported.
template <bool FP8, int MODE, bool DENSE, bool SPLIT, int RL, bool TWO, int BMT, bool PM = false, bool PAIRED = false>
__global__ void __launch_bounds__(THREADS, 1) routed_fused_kernel(const Params p) {
    static_assert(!DENSE || MODE == 2, "the dense case is the single-projection (down) mode");
    static_assert(!SPLIT || DENSE, "a K split is a dense scheduling device");
    // PM is only ever instantiated for the one-run rate-4 routed case; the
    // assert pins that so a future call cannot quietly read a wrong layout.
    // The piece-major reader is the one-run rate-4 ROUTED body: gate/up
    // (MODE 0/1) or the routed down (MODE 2, not the DENSE flag).  It is the
    // E4M3 MMA reader only.
    static_assert(!PM || (!TWO && RL == 4 && !DENSE && FAMILY_MMA8 && FP8),
                  "the piece-major reader is the E4M3 one-run rate-4 routed body only");
    // Paired is the LEGACY resident words only: the piece-major reader never
    // pairs, and the specialization is named only inside its matching
    // family/scope.
    static_assert(!PAIRED || (!PM && FP8 && FAMILY_MMA8 && !DENSE && !SPLIT
                  && (MODE == 0 || MODE == 2) && RL == 4 && !TWO && BMT == BM_WIDE),
                  "paired specialization is legacy words only, named only inside its matching family/scope");
    static_assert(launch_decodes(MODE, RL, TWO, DENSE), "only the pairs the launch decodes are instantiated");
    static_assert(has_width(FP8, FAMILY_MMA8, MODE, BMT) && !(SPLIT && BMT != BM),
                  "wide superblocks: the launches ``has_width`` names, unsplit");
    static_assert(launch_smem_bytes(MODE, pair_slot_words(RL, TWO), BMT, PAIRED) <= SM121_SMEM_OPTIN,
                  "the pair fits the target's block at this width");
    using L = Layout<MODE, BMT, PAIRED>;
    constexpr int PREFETCH_DISTANCE = FAMILY_MMA8 ? A_PREFETCH
        : (!FAMILY_FP8 && !FAMILY_FP4 ? VALUE_A_PREFETCH : 0);
    // The activation ring (MMA8_A_RING): the routed two-run launches.
    constexpr bool A_RING = MMA8_A_RING && TWO && !DENSE;
    constexpr bool PREFETCH_A = PREFETCH_DISTANCE > 0 && !DENSE && !TWO && !A_RING;
    // The E4M3 libraries' dense launch: a module's roles in one launch, its K
    // split reduced in-kernel (``DenseRoles``, ``fixup``).  The value family's
    // dense launch and every routed launch keep their code.
    constexpr bool MULTI = DENSE && FP8;
    // #793: only single-run MMA8 launches stage stream history.
    // Keep PREV_REGION_BYTES allocated so layout/ABI and bank mapping stay fixed.
    constexpr bool STAGE_PREV = PREV_STAGED && !TWO;
    // One A tile of BMT rows, and the consumers' rows: two warp rows of
    // BMT / 2, in MI blocks of 16.  E4M3: threads 0 .. 2 * BMT - 1 stage the
    // tile, two per row.
    constexpr int A_STAGE = BMT * BK * ELEM_BYTES;
    constexpr int WROWS = BMT / 2;
    constexpr int MI = WROWS / 16;
    static_assert(!FP8 || 2 * BMT <= PRODUCER_THREADS, "two producers per E4M3 A row");
    static_assert(FP8 || BMT == BM, "the value family stages 64 routes");
    extern __shared__ __align__(128) uint8_t smem[];
    TabT* tab = reinterpret_cast<TabT*>(smem + L::OFF_TABLES);
    uint8_t* Bs = smem + L::OFF_B;
    uint8_t* As = smem + L::OFF_A;
    uint8_t* Ar = As + STAGES * A_STAGE;    // the activation ring (A_RING only)
    int32_t* Ws = reinterpret_cast<int32_t*>(smem + L::OFF_W);
    int32_t* Ps = reinterpret_cast<int32_t*>(smem + L::OFF_PREV);   // STAGE_PREV only
    float* wsc = reinterpret_cast<float*>(smem + L::OFF_WSCALE);
    int32_t* desc = reinterpret_cast<int32_t*>(smem + L::OFF_DESC);
    int32_t* claim = reinterpret_cast<int32_t*>(smem + L::OFF_CLAIM);

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int nk = p.K / BK;
    const int dense_nsb = (p.rows_x + BMT - 1) / BMT;    // DENSE: superblocks of x
    const int total_items = DENSE ? dense_nsb * p.k_split * p.n_blocks
                                  : p.item_off[p.E] * p.n_blocks;
    unsigned gc = 0;          // global chunk counter: the stage is gc & 1
                              // (paired: one quantum per ++gc, two K32 microsteps)
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
                // FULL(gc) reuses FULL(gc-2), just like an ordinary chunk.
                // The last real chunk waited only for EMPTY(gc-3); producers
                // may still be ahead of the consumers by two chunks (#855).
                if (gc >= 2) bar_sync(BAR_EMPTY0 + (gc & 1), THREADS);
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
                if constexpr (MULTI) {
                    // nb is the block in the launch's list: find its role
                    #pragma unroll
                    for (int r = 1; r < MAX_ROLES; ++r)
                        if (r < p.roles.n && nb >= p.roles.block0[r]) e = r;
                    nb -= role_pick(p.roles.block0, e);
                }
                pos0 = sb * BMT;
                mb = min(BMT, p.rows_x - pos0);
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
                pos0 = start + sb * BMT;           // item_off counts BMT-route superblocks
                mb = min(BMT, end - pos0);
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
                    if constexpr (MULTI) {
                        v = (n0 + c < role_pick(p.roles.rows, e)) ? role_pick(p.roles.wscale, e)[n0 + c] : 0.f;
                    } else if constexpr (DENSE) {
                        // N-tail: a role's last block may hold fewer than BN
                        // rows; the columns past N are never stored.
                        v = (n0 + c < p.N) ? p.wscale0[(long)e * p.N + n0 + c] : 0.f;
                    } else {
                        v = p.wscale0[(long)e * p.N + n0 + c];
                    }
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
                constexpr int PER16 = 16 / ELEM_BYTES;   // table entries per 16-byte copy
                const TabT* t0 = MULTI ? static_cast<const TabT*>(role_pick(p.roles.table, e))
                                       : static_cast<const TabT*>(p.table0) + (long)e * TABLE_ENTRIES;
                for (int i = tid; i < TABLE_ENTRIES / PER16; i += PRODUCER_THREADS)
                    cp_async16(tab + i * PER16, t0 + i * PER16);
                if (MODE != 2) {
                    const TabT* t1 = static_cast<const TabT*>(p.table1) + (long)e * TABLE_ENTRIES;
                    for (int i = tid; i < TABLE_ENTRIES / PER16; i += PRODUCER_THREADS)
                        cp_async16(tab + TABLE_ENTRIES + i * PER16, t1 + i * PER16);
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
                if constexpr (MULTI) {
                    tbase_h[h] = role_pick(p.roles.words, e) + (long)g * p.tile_words;
                    init_h[h] = role_pick(p.roles.init, e);
                    hasinit_h[h] = role_pick(p.roles.has_init, e)[0];
                    rp_h[h] = load_runs(role_pick(p.roles.runs, e), 0);
                    bdesc_h[h] = role_pick(p.roles.bdesc, e);
                } else {
                    // The tile origin is the same in both layouts: piece 0 starts
                    // at the tile, and piece k is an offset INSIDE it (a piece
                    // plane is tile_words / PIECES_PER_TILE long).  Only the three
                    // inner addresses below move, never this base.
                    tbase_h[h] = (second ? p.words1 : p.words0) + (long)e * p.words_stride
                                 + (long)g * p.tile_words;
                    init_h[h] = (second ? p.init1 : p.init0) + (long)e * p.K;
                    hasinit_h[h] = (second ? p.has_init1 : p.has_init0)[e];
                    rp_h[h] = load_runs(second ? p.runs1 : p.runs0, e);
                    bdesc_h[h] = (second ? p.bdesc1 : p.bdesc0) + (long)e * nk * BDESC_INTS;
                }
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
                // A piece-major launch is only ever the one-run rate-4 case;
                // any other shape with the flag set is a caller bug, not a
                // decode to attempt.  (Routed only: DENSE never sets the flag.)
                if (PM && (TWO || RL != 4 || DENSE)) __trap();
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
            // E4M3: threads 0 .. 2 * BMT - 1, two per row; value family: every
            // producer, four per row.
            long arow = -1;
            {
                const int r = FP8 ? (tid >> 1) : (tid >> 2);
                if ((!FP8 || tid < 2 * BMT) && r < mb) {
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
            // ... and where its stream history comes from (STAGE_PREV).
            const int g_i = ih ? g_h[1] : g_h[0];
            const int32_t* init_i = ih ? init_h[1] : init_h[0];
            const int hasinit_i = ih ? hasinit_h[1] : hasinit_h[0];
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
            auto prefetch_a = [&](int kc) {
                if (arow >= 0) {
                    if constexpr (FP8)
                        prefetch_l1(reinterpret_cast<const uint8_t*>(p.x) + arow * p.K + kc * BK + (tid & 1) * 16);
                    else
                        prefetch_l1(reinterpret_cast<const uint16_t*>(p.x) + arow * p.K + kc * BK + (tid & 3) * 8);
                }
            };
            auto store_a = [&](int stage, const uint4& a) {
                uint8_t* A = As + stage * A_STAGE;
                if constexpr (FAMILY_MMA8) {
                    // E4M3 bytes as they are, in the MMA's k order (see E4M3
                    // MMA LAYOUT): word q of the 16-byte half holds physical k
                    // (2q, 2q + 1, 2q + 8, 2q + 9).
                    if (tid >= 2 * BMT) return;
                    const int row = tid >> 1;
                    const int c16 = tid & 1;
                    uint4 v = make_uint4(0, 0, 0, 0);
                    if (arow >= 0) {
                        v.x = __byte_perm(a.x, a.z, 0x5410); v.y = __byte_perm(a.x, a.z, 0x7632);
                        v.z = __byte_perm(a.y, a.w, 0x5410); v.w = __byte_perm(a.y, a.w, 0x7632);
                    }
                    *reinterpret_cast<uint4*>(A + row * BK + ((c16 ^ ((row >> 2) & 1)) << 4)) = v;
                } else if constexpr (FP8) {
                    if (tid >= 2 * BMT) return;
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
                // The word stages this pair's launch cycles through
                // (``word_stages``): at WORD_STAGES a chunk's words are
                // issued two chunks ahead of its decode, at WORD_STAGES_MIN
                // one.
                constexpr int WS = word_stages(MODE, SW);
                auto word_slot = [&](int kc) {
                    if constexpr (PAIRED) return ((kc / 2) & 1) * 2 + (kc & 1);
                    else return kc % WS;
                };
                static_assert(WS == WORD_STAGES || WS == WORD_STAGES_MIN, "two or three word stages");
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
                // Chunk kc's 16 raw activation bytes of this thread's A row
                // into its own slot of the activation ring (A_RING; see
                // MMA8_A_RING), in the commit group of chunk kc's words.  The
                // slot is the words' stage, so it is free once this thread's
                // ``store_a`` of chunk kc - WS has read it.
                static_assert(!A_RING || WS == WORD_STAGES, "the activation ring rides three word stages");
                auto issue_a = [&](int kc) {
                    if constexpr (A_RING) {
                        if (arow >= 0)
                            cp_async16(Ar + (kc % WS) * A_STAGE + tid * 16,
                                       reinterpret_cast<const uint8_t*>(p.x) + arow * p.K + kc * BK + (tid & 1) * 16);
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
                        // Legacy: a column's chunk starts at cw0 and its t64-th
                        // piece is 2*rate words in.  Piece-major: the piece
                        // planes are tile_words / PIECES_PER_TILE long, and the
                        // column's own words sit cw0 / PIECES_PER_TILE in.
                        const int32_t* src = PM
                            ? tbase_i + t64_i * (p.tile_words / PIECES_PER_TILE)
                                  + c.cw0 / PIECES_PER_TILE
                            : tbase_i + c.cw0 + 2 * c.rate * t64_i;
                        int32_t* dst = Ws + word_slot(kc) * W_STAGE + (ih * BK + mm) * SW;
                        if constexpr (TWO) {
                            if (c.lo) copy_half<RL>(dst, src, t64_i, q);
                            else copy_half<RH>(dst, src, t64_i, q);
                        } else {
                            copy_half<RL>(dst, src, t64_i, q);
                        }
                        // The staged stream history: the 32 bits before the
                        // half, in the same commit group as its words (see
                        // STAGE_PREV and ``load_prev``, whose cases these are).
                        if constexpr (STAGE_PREV) {
                            if (q == 1) {
                                int32_t* pd = Ps + word_slot(kc) * PREV_STAGE_INTS + ih * BK + mm;
                                if constexpr (PM) {
                                    // The word before the half in the piece
                                    // plane: the same column's previous piece's
                                    // last word; at t64 == 0 the previous tile's
                                    // piece 7, one plane stride below the first
                                    // piece (7*ps - tile_words == -ps).
                                    const int ps = p.tile_words / PIECES_PER_TILE;
                                    const int colbase = c.cw0 / PIECES_PER_TILE;
                                    if (t64_i > 0)
                                        cp_async4(pd, tbase_i + (t64_i - 1) * ps + colbase
                                                          + 2 * c.rate - 1);
                                    else if (g_i > 0)
                                        cp_async4(pd, tbase_i + colbase + 2 * c.rate - 1 - ps);
                                    else if (hasinit_i) cp_async4(pd, init_i + c.p);
                                    else *pd = 0;
                                } else {
                                    const int wr0 = 2 * c.rate * t64_i;
                                    const int32_t* wcol = tbase_i + c.cw0;
                                    if (wr0 > 0) cp_async4(pd, wcol + wr0 - 1);
                                    else if (g_i > 0) cp_async4(pd, wcol + 16 * c.rate - 1 - p.tile_words);
                                    else if (hasinit_i) cp_async4(pd, init_i + c.p);
                                    else *pd = 0;
                                }
                            }
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
                // With STAGE_PREV the word comes from its stage instead
                // (``issue_words``; read in the chunk body) and this maps the
                // columns only.
                auto load_prev = [&](int kc, int32_t (&pv)[2], ColMap (&cm)[2], bool ring) {
                    #pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        // MODE 2 reads one projection: both halves map alike.
                        cm[h] = (MODE == 2 && h == 1) ? cm[0] : col_map<RL, TWO>(blk_of(kc, h, ring), n_lo, w_hi, kc, m);
                        const ColMap& c = cm[h];
                        if (STAGE_PREV || 8 * j * c.rate >= 32) continue;
                        int32_t v;
                        if constexpr (PM) {
                            const int ps = p.tile_words / PIECES_PER_TILE;
                            const int colbase = c.cw0 / PIECES_PER_TILE;
                            const int t64 = t64_h[h];
                            if (t64 > 0) v = tbase_h[h][(t64 - 1) * ps + colbase + 2 * c.rate - 1];
                            else if (g_h[h] > 0) v = tbase_h[h][colbase + 2 * c.rate - 1 - ps];
                            else v = hasinit_h[h] ? init_h[h][c.p] : 0;
                        } else {
                            const int wr0 = 2 * c.rate * t64_h[h];
                            const int32_t* wcol = tbase_h[h] + c.cw0;
                            if (wr0 > 0) v = wcol[wr0 - 1];
                            else if (g_h[h] > 0) v = wcol[16 * c.rate - 1 - p.tile_words];
                            else v = hasinit_h[h] ? init_h[h][c.p] : 0;
                        }
                        pv[h] = v;
                    }
                };

                if constexpr (PAIRED) {
                    // One-run schedule: both K32 chunks' words and history in
                    // ONE commit group with the tables; the quantum loop's
                    // single wait retires it before the first decode.
                    issue_words(0, false);
                    issue_words(1, false);
                    cp_async_commit();  // tables + both K32 words/history, ONE group
                } else {
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
                    //
                    // At WORD_STAGES_MIN group 1 carries no words, and the group
                    // iteration kc commits carries chunk kc + 1's words (and chunk
                    // kc + 4's descriptors); the iteration waits for every group,
                    // so the same visibility holds a fortiori: chunk kc + 1's
                    // descriptors, which ``issue_words(kc + 1)`` maps from, joined
                    // a group committed no later than iteration kc - 3 (group 0
                    // for kc0 + 2, group 1 for kc0 + 3; kc0 + 1's are stored
                    // directly), and the overwrite of chunk kc's slot still
                    // follows its last reader (``issue_words`` at iteration
                    // kc - 1, ``load_prev`` at kc - 1) across iteration kc's
                    // barrier.
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
                    issue_a(kc0);
                    if (nkc > 2) issue_desc(kc0 + 2);
                    cp_async_commit();                     // group 0
                    if constexpr (WS == WORD_STAGES) {
                        if (nkc > 1) { issue_words(kc0 + 1, false); issue_a(kc0 + 1); }
                    }
                    if (nkc > 3) issue_desc(kc0 + 3);
                    cp_async_commit();                     // group 1
                    if constexpr (TWO) bar_sync(BAR_PROD, PRODUCER_THREADS);   // chunks kc0, kc0 + 1's descriptors
                }
                int32_t prev_cur[2] = {0, 0}, prev_nxt[2] = {0, 0};
                ColMap cm_cur[2], cm_nxt[2];
                uint4 a_cur = make_uint4(0, 0, 0, 0), a_nxt = make_uint4(0, 0, 0, 0);
                load_prev(kc0, prev_cur, cm_cur, TWO);
                if constexpr (!A_RING) load_a(kc0, a_cur);
                if constexpr (PREFETCH_A) {
                    #pragma unroll
                    for (int d = 2; d < PREFETCH_DISTANCE; ++d)
                        if (d < nkc) prefetch_a(kc0 + d);
                }
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
                    if constexpr (!STAGE_PREV) { prev_cur[0] ^= zero; prev_cur[1] ^= zero; }
                    if constexpr (!A_RING) {
                        a_cur.x ^= (uint32_t)zero; a_cur.y ^= (uint32_t)zero;
                        a_cur.z ^= (uint32_t)zero; a_cur.w ^= (uint32_t)zero;
                    }
                }
                // Ordinary K32 decode/store, shared verbatim by both schedules.
                auto publish_micro = [&](int kc, int stage) {
                    store_a(stage, a_cur);
                    const int32_t* W = Ws + word_slot(kc) * W_STAGE;
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
                    // The chunk's stream history: from its stage (STAGE_PREV;
                    // zero for the row groups ``load_prev`` skips, as the
                    // carried registers held), or carried from ``load_prev``.
                    int32_t pv[2];
                    #pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        if constexpr (STAGE_PREV)
                            pv[h] = (8 * j * cm_cur[h].rate < 32) ? Ps[word_slot(kc) * PREV_STAGE_INTS + h * BK + m] : 0;
                        else
                            pv[h] = prev_cur[h];
                    }
                    const TabT* T0 = tab;
                    const TabT* T1 = tab + ((MODE == 2) ? 0 : TABLE_ENTRIES);
                    uint32_t packed[2][4];
                    if constexpr (!TWO) {
                        decode_two<FP8, RL, RL>(Wc, pv, j, T0, T1, ws, packed);
                    } else if constexpr (MODE == 2) {
                        // one column in both halves: one rate
                        if (cm_cur[0].lo) decode_two<FP8, RL, RL>(Wc, pv, j, T0, T1, ws, packed);
                        else decode_two<FP8, RH, RH>(Wc, pv, j, T0, T1, ws, packed);
                    } else {
                        // gate and up share the pair, not the column order
                        const bool lo0 = cm_cur[0].lo, lo1 = cm_cur[1].lo;
                        if (lo0 && lo1) decode_two<FP8, RL, RL>(Wc, pv, j, T0, T1, ws, packed);
                        else if (!lo0 && !lo1) decode_two<FP8, RH, RH>(Wc, pv, j, T0, T1, ws, packed);
                        else if (lo0) decode_two<FP8, RL, RH>(Wc, pv, j, T0, T1, ws, packed);
                        else decode_two<FP8, RH, RL>(Wc, pv, j, T0, T1, ws, packed);
                    }
                    #pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int cib = cm_cur[h].cib;
                        if constexpr (FAMILY_MMA8)
                            *reinterpret_cast<uint2*>(B + cib * BN + b8off(chunk[h], cib)) =
                                make_uint2(packed[h][0], packed[h][1]);
                        else
                            *reinterpret_cast<uint4*>(B + cib * (BN * 2) + (bswz(chunk[h], cib) << 4)) =
                                make_uint4(packed[h][0], packed[h][1], packed[h][2], packed[h][3]);
                    }
                };
                auto advance_micro = [&]() {
                    if constexpr (!STAGE_PREV) { prev_cur[0] = prev_nxt[0]; prev_cur[1] = prev_nxt[1]; }
                    cm_cur[0] = cm_nxt[0]; cm_cur[1] = cm_nxt[1];
                    if constexpr (!A_RING) a_cur = a_nxt;
                };
                if constexpr (PAIRED) {
                    const int np = nkc / 2;
                    for (int q = 0; q < np; ++q, ++gc) {
                        const int kc = 2 * q;
                        // Preserve exactly one current/next register pipeline.
                        load_prev(kc + 1, prev_nxt, cm_nxt, false);
                        load_a(kc + 1, a_nxt);
                        if constexpr (PREFETCH_A) {
                            if (kc + A_PREFETCH < nkc) prefetch_a(kc + A_PREFETCH);
                        }
                        cp_async_wait<0>();
                        bar_sync(BAR_PROD, PRODUCER_THREADS);
                        // Pair q-1's producer reads retired at the barrier;
                        // pair q+1 reuses its word slot, never current pair q's.
                        if (q + 1 < np) {
                            issue_words(kc + 2, false);
                            issue_words(kc + 3, false);
                        }
                        cp_async_commit();  // ONE next pair, or a uniform empty group
                        const int stage = gc & 1;
                        if (gc >= 2) bar_sync(BAR_EMPTY0 + stage, THREADS);
                        publish_micro(kc, 2 * stage);
                        advance_micro();
                        if (kc + 2 < nkc) {
                            load_prev(kc + 2, prev_nxt, cm_nxt, false);
                            load_a(kc + 2, a_nxt);
                        }
                        if constexpr (PREFETCH_A) {
                            if (kc + 1 + A_PREFETCH < nkc) prefetch_a(kc + 1 + A_PREFETCH);
                        }
                        publish_micro(kc + 1, 2 * stage + 1);
                        advance_micro();
                        bar_arrive(BAR_FULL0 + stage, THREADS);
                    }
                    // Retire the final empty commit before the next item queues
                    // its LUT and words. Its BAR_PROD precedes all reuse.
                    cp_async_wait<0>();
                } else {
                    for (int ic = 0; ic < nkc; ++ic, ++gc) {
                        const int kc = kc0 + ic;
                        // The two orders are the measured ones: a one-run loop that
                        // issues the activation chunk first waits longer at M = 1.
                        if constexpr (A_RING) {
                            if (ic + 1 < nkc) load_prev(kc + 1, prev_nxt, cm_nxt, TWO);
                        } else if constexpr (TWO) {
                            if (ic + 1 < nkc) { load_a(kc + 1, a_nxt); load_prev(kc + 1, prev_nxt, cm_nxt, true); }
                        } else {
                            if (ic + 1 < nkc) { load_prev(kc + 1, prev_nxt, cm_nxt, false); load_a(kc + 1, a_nxt); }
                        }
                        if constexpr (PREFETCH_A) {
                            if (ic + PREFETCH_DISTANCE < nkc) prefetch_a(kc + PREFETCH_DISTANCE);
                        }
                        // Chunk kc's words (and the tables) have landed ...
                        if constexpr (WS == WORD_STAGES) cp_async_wait<1>();
                        else cp_async_wait<0>();
                        // A_RING: chunk kc's activation bytes landed with its words,
                        // and this thread copied them, so they are visible to it
                        // now; read here, the load's latency hides behind the barrier.
                        if constexpr (A_RING) {
                            if (arow >= 0) a_cur = *reinterpret_cast<const uint4*>(Ar + (kc % WS) * A_STAGE + tid * 16);
                        }
                        bar_sync(BAR_PROD, PRODUCER_THREADS);   // ... for every producer; chunk kc-1's stage is free
                        if constexpr (WS == WORD_STAGES) {
                            if (ic + 2 < nkc) { issue_words(kc + 2, TWO); issue_a(kc + 2); }
                        } else {
                            if (ic + 1 < nkc) issue_words(kc + 1, TWO);
                        }
                        if (ic + 4 < nkc) issue_desc(kc + 4);
                        cp_async_commit();
                        const int stage = gc & 1;
                        if (gc >= 2) bar_sync(BAR_EMPTY0 + stage, THREADS);
                        publish_micro(kc, stage);
                        bar_arrive(BAR_FULL0 + stage, THREADS);
                        advance_micro();
                    }
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
        const int mw = cw >> 2;      // WROWS rows
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
            // MULTI: the item's role's rows and first output column; one role
            // at column 0 otherwise.
            int n_role = p.N, col0 = 0;
            if constexpr (MULTI) {
                n_role = role_pick(p.roles.rows, e);
                col0 = role_pick(p.roles.col, e);
            }
            // This warp's 16-row blocks that hold a route.  On the wide
            // superblock a block wholly past ``mb`` is padding: its rows are
            // never written (the epilogue's ``r >= mb``), so its fragment
            // loads and MMAs are skipped.  ``mb`` is the item's, so the branch
            // is warp-uniform around ``mma.sync``, and a live row's
            // accumulator sees the same MMAs in the same order either way.
            // The 64-route launch keeps its published code (every block live).
            const int live = (BMT == BM) ? MI : min(MI, max(0, (mb - WROWS * mw + 15) >> 4));
            float acc[MI][4][4];
            #pragma unroll
            for (int mi = 0; mi < MI; ++mi)
                #pragma unroll
                for (int nt = 0; nt < 4; ++nt)
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) acc[mi][nt][i] = 0.f;
            for (int ic = 0; ic < nkc / L::MICROS; ++ic, ++gc) {
                stage = gc & 1;
                if (ic > 0) bar_sync(BAR_FULL0 + stage, THREADS);
                #pragma unroll
                for (int u = 0; u < L::MICROS; ++u) {
                    const int micro = stage * L::MICROS + u;
                    const uint8_t* A = As + micro * A_STAGE;
                    const uint8_t* B = Bs + micro * B_STAGE_BYTES;
                    if constexpr (FAMILY_MMA8) {
                        // One k32 step (see E4M3 MMA LAYOUT).  acc[mi][2G + parity]:
                        // column group G's even-n (0) and odd-n (1) MMA.
                        uint32_t a[MI][4];
                        #pragma unroll
                        for (int mi = 0; mi < MI; ++mi) {
                            if (mi >= live) break;
                            const int row = WROWS * mw + 16 * mi + 8 * ((lane >> 3) & 1) + (lane & 7);
                            ldmatrix_x4(a[mi], A + row * BK + (((lane >> 4) ^ ((row >> 2) & 1)) << 4));
                        }
                        auto load_b = [&](int G, uint32_t (&be)[2], uint32_t (&bo)[2]) {
                            uint32_t X[4];
                            // lane l addresses row l of matrix l >> 3: k = lane
                            ldmatrix_x4_trans(X, B + lane * BN + ((((2 * nw + G) ^ b8swz(lane)) & 7) << 4));
                            be[0] = __byte_perm(X[0], X[1], 0x6420);
                            be[1] = __byte_perm(X[2], X[3], 0x6420);
                            bo[0] = __byte_perm(X[0], X[1], 0x7531);
                            bo[1] = __byte_perm(X[2], X[3], 0x7531);
                        };
                        if constexpr (MMA8_GATE_UP_B_PREFETCH && MODE == 0 && !DENSE && !TWO && RL == 4) {
                            uint32_t be[2][2], bo[2][2];
                            #pragma unroll
                            for (int G = 0; G < 2; ++G) load_b(G, be[G], bo[G]);
                            #pragma unroll
                            for (int G = 0; G < 2; ++G) {
                                #pragma unroll
                                for (int mi = 0; mi < MI; ++mi) {
                                    if (mi >= live) break;
                                    mma16832_e4m3(acc[mi][2 * G], a[mi], be[G]);
                                    mma16832_e4m3(acc[mi][2 * G + 1], a[mi], bo[G]);
                                }
                            }
                        } else {
                            #pragma unroll
                            for (int G = 0; G < 2; ++G) {
                                uint32_t be[2], bo[2];
                                load_b(G, be, bo);
                                #pragma unroll
                                for (int mi = 0; mi < MI; ++mi) {
                                    if (mi >= live) break;
                                    mma16832_e4m3(acc[mi][2 * G], a[mi], be);
                                    mma16832_e4m3(acc[mi][2 * G + 1], a[mi], bo);
                                }
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int s = 0; s < BK / 16; ++s) {
                            uint32_t a[MI][4];
                            const int q = lane >> 3;
                            #pragma unroll
                            for (int mi = 0; mi < MI; ++mi) {
                                if (mi >= live) break;
                                const int row = WROWS * mw + 16 * mi + 8 * (q & 1) + (lane & 7);
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
                            for (int mi = 0; mi < MI; ++mi) {
                                if (mi >= live) break;
                                #pragma unroll
                                for (int nt = 0; nt < 4; ++nt) mma16816<FP8>(acc[mi][nt], a[mi], b[nt]);
                            }
                        }
                    }
                }  // original per-accumulator K32 order: u0, then u1
                bar_arrive(BAR_EMPTY0 + stage, THREADS);
            }
            // ------------------------------------------------------ epilogue
            // A thread's columns in one row come in NSEG segments of SEGW
            // consecutive B columns: four pairs at 32 nw + 8 s + 2 q on the
            // f16 instruction, two quads at 32 nw + 16 s + 4 q on the E4M3
            // one (see E4M3 MMA LAYOUT).  ``accv(mi, s, c, hr)`` is element c
            // of segment s in row half hr.  Gate/up: the first NSEG / 2
            // segments are gate columns, the rest the up columns 16 on.
            constexpr int NSEG = FAMILY_MMA8 ? 2 : 4;
            constexpr int SEGW = FAMILY_MMA8 ? 4 : 2;
            static_assert(!DENSE || DENSE_ROW_QUANTUM % SEGW == 0,
                          "a dense store segment lies wholly inside the role or wholly past it");
            constexpr int SEGSTRIDE = FAMILY_MMA8 ? 16 : 8;
            const int q4 = lane & 3;
            auto accv = [&](int mi, int sg, int c, int hr) -> float {
                if constexpr (FAMILY_MMA8) return acc[mi][2 * sg + (c & 1)][2 * hr + (c >> 1)];
                else return acc[mi][sg][2 * hr + c];
            };
            // SEGW bf16 at a SEGW-aligned column, as SEGW / 2 four-byte
            // stores: both instructions need only the even row stride and
            // 4-byte base the host checks, so the two E4M3 libraries serve
            // the same output views.
            auto store_seg = [&](uint16_t* dst, const uint32_t (&w)[SEGW / 2]) {
                #pragma unroll
                for (int i = 0; i < SEGW / 2; ++i) reinterpret_cast<uint32_t*>(dst)[i] = w[i];
            };
            const int n0 = (MODE == 2) ? nb * BN : nb * HALF;
            uint16_t* out = reinterpret_cast<uint16_t*>(p.out);
            #pragma unroll
            for (int mi = 0; mi < MI; ++mi) {
                #pragma unroll
                for (int hr = 0; hr < 2; ++hr) {
                    const int r = WROWS * mw + 16 * mi + 8 * hr + (lane >> 2);
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
                        float* part = p.partial + ((long)ks * p.rows_x + pos) * p.N + col0 + n0;
                        #pragma unroll
                        for (int sg = 0; sg < NSEG; ++sg) {
                            const int cb = 32 * nw + SEGSTRIDE * sg + SEGW * q4;
                            // N-tail (SPLIT is dense-only): N % SEGW == 0, so a
                            // segment is wholly inside the role or wholly past it.
                            if (n0 + cb >= n_role) continue;
                            if constexpr (SEGW == 4)
                                *reinterpret_cast<float4*>(part + cb) = make_float4(
                                    accv(mi, sg, 0, hr), accv(mi, sg, 1, hr), accv(mi, sg, 2, hr), accv(mi, sg, 3, hr));
                            else
                                *reinterpret_cast<float2*>(part + cb) = make_float2(accv(mi, sg, 0, hr), accv(mi, sg, 1, hr));
                        }
                    } else if constexpr (MODE == 0) {
                        // gate segment sg pairs with up segment sg + NSEG / 2 in the same registers
                        #pragma unroll
                        for (int sg = 0; sg < NSEG / 2; ++sg) {
                            const int cb = 32 * nw + SEGSTRIDE * sg + SEGW * q4;       // gate B column
                            uint32_t w[SEGW / 2] = {};
                            #pragma unroll
                            for (int c = 0; c < SEGW; ++c) {
                                float g = accv(mi, sg, c, hr);
                                float u = accv(mi, sg + NSEG / 2, c, hr);
                                if constexpr (FP8) {
                                    g = __fmul_rn(__fmul_rn(g, a_s), wsc[slot * BN + cb + c]);
                                    u = __fmul_rn(__fmul_rn(u, a_s), wsc[slot * BN + cb + 16 + c]);
                                }
                                // the bf16 GEMM output, widened for the fp32 activation
                                float gf = bf16_bits_to_f32(bf16_bits_rn(g));
                                float uf = bf16_bits_to_f32(bf16_bits_rn(u));
                                gf = fminf(gf, p.limit);
                                uf = fmaxf(fminf(uf, p.limit), -p.limit);
                                const float act = __fmul_rn(gf / (1.0f + expf(-gf)), uf);
                                w[c >> 1] |= (uint32_t)bf16_bits_rn(act) << (16 * (c & 1));
                            }
                            const long col = n0 + 16 * nw + SEGSTRIDE * sg + SEGW * q4;
                            store_seg(out + (long)pos * p.out_stride + col, w);
                        }
                    } else if constexpr (MODE == 1) {
                        #pragma unroll
                        for (int sg = 0; sg < NSEG; ++sg) {
                            const int cb = 32 * nw + SEGSTRIDE * sg + SEGW * q4;
                            const int h = sg / (NSEG / 2);
                            const int nl = 16 * nw + SEGSTRIDE * (sg % (NSEG / 2)) + SEGW * q4;
                            uint32_t w[SEGW / 2] = {};
                            #pragma unroll
                            for (int c = 0; c < SEGW; ++c) {
                                float y = accv(mi, sg, c, hr);
                                if constexpr (FP8) y = __fmul_rn(__fmul_rn(y, a_s), wsc[slot * BN + cb + c]);
                                w[c >> 1] |= (uint32_t)bf16_bits_rn(y) << (16 * (c & 1));
                            }
                            const long col = (long)h * p.inter + n0 + nl;
                            store_seg(out + (long)flat * p.out_stride + col, w);
                        }
                    } else {
                        #pragma unroll
                        for (int sg = 0; sg < NSEG; ++sg) {
                            const int cb = 32 * nw + SEGSTRIDE * sg + SEGW * q4;
                            if constexpr (DENSE) {
                                // N-tail: N % SEGW == 0, so a segment is
                                // wholly inside the role or wholly past it.
                                if (n0 + cb >= n_role) continue;
                            }
                            uint32_t w[SEGW / 2] = {};
                            #pragma unroll
                            for (int c = 0; c < SEGW; ++c) {
                                float y = accv(mi, sg, c, hr);
                                if constexpr (FP8) y = __fmul_rn(__fmul_rn(y, a_s), wsc[slot * BN + cb + c]);
                                if (!DENSE && p.mul_weight) y = __fmul_rn(y, rw);
                                w[c >> 1] |= (uint32_t)bf16_bits_rn(y) << (16 * (c & 1));
                            }
                            const long col = col0 + n0 + cb;
                            store_seg(out + (long)flat * p.out_stride + col, w);
                        }
                    }
                }
            }
            if constexpr (MULTI && SPLIT) {
                if (p.fixup) {
                    // The split fixup.  Every split of a tile has written its
                    // raw partial; the one that arrives last sums the S
                    // partials in split order from 0.f -- dense_reduce_kernel's
                    // operation order -- and applies the epilogue, so the
                    // output is bitwise the two-launch one.  ``wsc[slot]`` is
                    // still this item's: every item has at least STAGES + 1
                    // chunks (the host refuses a split that leaves fewer), so
                    // the producers cannot reach item + 2's slot before the
                    // consumers start item + 1.
                    __threadfence();
                    bar_sync(BAR_CONS, THREADS - PRODUCER_THREADS);
                    if (tid == PRODUCER_THREADS) {
                        const int sb = desc[slot * 8 + 2];
                        const int tile = (role_pick(p.roles.block0, e) + nb) * dense_nsb + sb;
                        const int before = atomicAdd(p.tile_sem + tile, 1);
                        __threadfence();
                        claim[1] = before == p.k_split - 1;
                    }
                    bar_sync(BAR_CONS, THREADS - PRODUCER_THREADS);
                    if (claim[1]) {
                        #pragma unroll
                        for (int mi = 0; mi < MI; ++mi) {
                            #pragma unroll
                            for (int hr = 0; hr < 2; ++hr) {
                                const int r = WROWS * mw + 16 * mi + 8 * hr + (lane >> 2);
                                if (r >= mb) continue;
                                const int pos = pos0 + r;
                                const float a_s = p.a_scale[pos];
                                #pragma unroll
                                for (int sg = 0; sg < NSEG; ++sg) {
                                    const int cb = 32 * nw + SEGSTRIDE * sg + SEGW * q4;
                                    if (n0 + cb >= n_role) continue;
                                    float y[SEGW];
                                    #pragma unroll
                                    for (int c = 0; c < SEGW; ++c) y[c] = 0.f;
                                    for (int s = 0; s < p.k_split; ++s) {
                                        const float* src = p.partial + ((long)s * p.rows_x + pos) * p.N + col0 + n0 + cb;
                                        if constexpr (SEGW == 4) {
                                            const float4 v = __ldcg(reinterpret_cast<const float4*>(src));
                                            y[0] += v.x; y[1] += v.y; y[2] += v.z; y[3] += v.w;
                                        } else {
                                            const float2 v = __ldcg(reinterpret_cast<const float2*>(src));
                                            y[0] += v.x; y[1] += v.y;
                                        }
                                    }
                                    uint32_t w[SEGW / 2] = {};
                                    #pragma unroll
                                    for (int c = 0; c < SEGW; ++c) {
                                        const float v = __fmul_rn(__fmul_rn(y[c], a_s), wsc[slot * BN + cb + c]);
                                        w[c >> 1] |= (uint32_t)bf16_bits_rn(v) << (16 * (c & 1));
                                    }
                                    store_seg(out + (long)pos * p.out_stride + col0 + n0 + cb, w);
                                }
                            }
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

// token_sum with the MoE runner's shared-expert add folded in
// (TESSERA_GLM53_FOLD_SHARED_ADD, tessera.serving.glm53_shared_fold):
// out[t, :] = bf16( f32(shared[t, :]) + f32(bf16( sum_{j < top_k} f32(routed[t * top_k + j, :]) )) ).
// The routed sum is rounded to bf16 exactly as token_sum_kernel stores it, then
// added the way ATen's bf16 ``shared + routed`` does: one fp32 add, rounded to
// nearest even by the same intrinsic.  Bitwise equal to token_sum_kernel
// followed by that add.
__global__ void token_sum_shared_kernel(const uint16_t* __restrict__ routed,
                                        const uint16_t* __restrict__ shared,
                                        uint16_t* __restrict__ out,
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
    const uint4 sv = *reinterpret_cast<const uint4*>(shared + t * width + c);
    const uint32_t sw[4] = {sv.x, sv.y, sv.z, sv.w};
    uint32_t ow[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float lo = __fadd_rn(bf16_bits_to_f32(sw[i] & 0xFFFFu),
                                   bf16_bits_to_f32(bf16_bits_rn(acc[2 * i])));
        const float hi = __fadd_rn(bf16_bits_to_f32(sw[i] >> 16),
                                   bf16_bits_to_f32(bf16_bits_rn(acc[2 * i + 1])));
        ow[i] = bf16_bits_rn(lo) | ((uint32_t)bf16_bits_rn(hi) << 16);
    }
    uint4 o;
    o.x = ow[0];
    o.y = ow[1];
    o.z = ow[2];
    o.w = ow[3];
    *reinterpret_cast<uint4*>(out + t * width + c) = o;
}

int max_dynamic_smem_bytes(int device) {
    int v = 0;
    TESSERA_HOST_CUDA(cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    return v;
}

template <bool FP8, int MODE, bool DENSE, bool SPLIT, int RL, bool TWO, int BMT, bool PM, bool PAIRED = false>
void launch_variant(const Params& p, int grid, cudaStream_t stream) {
    const int smem = launch_smem_bytes(MODE, p.slot_words, BMT, PAIRED);
    static int attributed = 0;     // the largest dynamic size this instantiation was granted
    if (smem > attributed) {
        TESSERA_HOST_CUDA(cudaFuncSetAttribute(routed_fused_kernel<FP8, MODE, DENSE, SPLIT, RL, TWO, BMT, PM, PAIRED>,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        attributed = smem;
    }
    routed_fused_kernel<FP8, MODE, DENSE, SPLIT, RL, TWO, BMT, PM, PAIRED><<<grid, THREADS, smem, stream>>>(p);
    TESSERA_HOST_CUDA(cudaGetLastError());
}

template <bool FP8, int MODE, bool DENSE, bool SPLIT, int RL, bool TWO, int BMT, bool PM = false>
void launch_pair(const Params& p, int grid, cudaStream_t stream) {
    // The paired schedule is admitted only inside its matching family/scope
    // and never on a piece-major launch (PM is compiled out above): a shape
    // the scope declines runs the ordinary specialization, unchanged.
    if constexpr (!PM && PAIRED_K32_BUILD && FP8 && FAMILY_MMA8 && !DENSE && !SPLIT
                  && (MODE == 0 || MODE == 2) && RL == 4 && !TWO && BMT == BM_WIDE) {
        if (paired_k32_scope(FP8, FAMILY_MMA8, MODE, DENSE, SPLIT, RL, TWO, BMT, p.K, p.rows_x, p.slot_words)) {
            launch_variant<FP8, MODE, DENSE, SPLIT, RL, TWO, BMT, PM, true>(p, grid, stream);
            return;
        }
    }
    launch_variant<FP8, MODE, DENSE, SPLIT, RL, TWO, BMT, PM>(p, grid, stream);
}

// The launch's run pair, from its ``tile_words`` (``pair_of``), picks the
// kernel instantiation.  A pair this launch does not decode
// (``launch_decodes``) is refused here, before any launch; the Python owner
// refuses it first (``fused_routed_window_supported``: the device's shared
// memory; ``run_pair``: adjacency).
template <bool FP8, int MODE, bool DENSE = false, bool SPLIT = false, int BMT = BM>
void launch(const Params& p, int grid, cudaStream_t stream) {
    const PairKey k = pair_of(p.tile_words, p.K);
    // One case per (r_lo, two); a two-run key is offset past every rate.
    constexpr int KEY_TWO = 16;
    static_assert(RATE_MAX < KEY_TWO, "one-run and two-run keys stay apart");
    // Piece-major is the E4M3 MMA one-run rate-4 ROUTED body (tessera#739):
    // gate/up MODE 0/1, or the routed down MODE 2 (DENSE is the separate dense
    // Linear flag, not MODE).  The guard is ``if constexpr`` so the PM
    // instantiation is never even named for another family or a dense launch;
    // a request there is refused, never silently decoded as legacy.
    TESSERA_HOST_CHECK(!p.piece_major || (FAMILY_MMA8 && FP8), "the piece-major reader is the E4M3 MMA reader only");
    if constexpr (FAMILY_MMA8 && FP8 && !DENSE) {
        if (p.piece_major) {
            TESSERA_HOST_CHECK(!k.two && k.r_lo == 4, "the piece-major reader is the one-run rate-4 routed body only");
            if constexpr (launch_decodes(MODE, 4, false, DENSE)) {
                launch_pair<FP8, MODE, false, SPLIT, 4, false, BMT, true>(p, grid, stream);
                return;
            }
            TESSERA_HOST_CHECK(false, "no piece-major rate-4 launch for this mode/width");
        }
    }
    switch (k.two ? KEY_TWO + k.r_lo : k.r_lo) {
#define TESSERA_ROUTED_FUSED_PAIR(R, T)                                                        \
        case (T ? KEY_TWO : 0) + R:                                                            \
            if constexpr (launch_decodes(MODE, R, T, DENSE)) {                                 \
                launch_pair<FP8, MODE, DENSE, SPLIT, R, T, BMT>(p, grid, stream);              \
                return;                                                                        \
            }                                                                                  \
            break;
        TESSERA_ROUTED_FUSED_PAIR(1, false) TESSERA_ROUTED_FUSED_PAIR(2, false)
        TESSERA_ROUTED_FUSED_PAIR(3, false) TESSERA_ROUTED_FUSED_PAIR(4, false)
        TESSERA_ROUTED_FUSED_PAIR(5, false) TESSERA_ROUTED_FUSED_PAIR(6, false)
        TESSERA_ROUTED_FUSED_PAIR(7, false) TESSERA_ROUTED_FUSED_PAIR(8, false)
        TESSERA_ROUTED_FUSED_PAIR(9, false) TESSERA_ROUTED_FUSED_PAIR(10, false)
        TESSERA_ROUTED_FUSED_PAIR(11, false) TESSERA_ROUTED_FUSED_PAIR(12, false)
        TESSERA_ROUTED_FUSED_PAIR(13, false) TESSERA_ROUTED_FUSED_PAIR(14, false)
        TESSERA_ROUTED_FUSED_PAIR(1, true) TESSERA_ROUTED_FUSED_PAIR(2, true)
        TESSERA_ROUTED_FUSED_PAIR(3, true) TESSERA_ROUTED_FUSED_PAIR(4, true)
        TESSERA_ROUTED_FUSED_PAIR(5, true) TESSERA_ROUTED_FUSED_PAIR(6, true)
        TESSERA_ROUTED_FUSED_PAIR(7, true) TESSERA_ROUTED_FUSED_PAIR(8, true)
        TESSERA_ROUTED_FUSED_PAIR(9, true) TESSERA_ROUTED_FUSED_PAIR(10, true)
        TESSERA_ROUTED_FUSED_PAIR(11, true) TESSERA_ROUTED_FUSED_PAIR(12, true)
        TESSERA_ROUTED_FUSED_PAIR(13, true)
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

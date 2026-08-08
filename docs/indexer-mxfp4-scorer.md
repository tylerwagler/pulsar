# Indexer scorer: native block-scaled MXFP4 design

Design note, not yet implemented. Target: replace
`indexer_scores_wmma128_kernel` (92.38 ms of a 4096-token prefill) with a
block-scaled MMA kernel that consumes the compressed cache in its stored
format.

## Why the current kernel is slow

It is not math-bound. Per block it does 64 serial head iterations, each
staging a 16x128 Q tile into shared, running one 16x16x16 WMMA chain over
k=128 (eight `mma_sync`), storing to `c_sh` and reducing — with two
`__syncthreads()` per head, **128 per block**. The arithmetic per iteration
(16x128x128) is trivial next to that overhead.

Two structural costs dominate:

1. **Dequantisation.** With `PULSAR_IDX_FP4` on (default), the compressed rows
   are stored MXKV-FP4-packed at 68 B/row (64 B of E2M1 nibbles, low-nibble
   first, plus 4 E8M0 block-32 scales). `idx_comp_load_dev` unpacks every
   nibble, applies the E8 scale, returns f32 — and the kernel then converts to
   `__half` into a 32 KB `b_sh`. We pay to unpack the exact format the tensor
   cores could consume directly.
2. **Occupancy.** Grid is `((n_comp+127)/128, (n_tokens+15)/16)`. `n_comp`
   ramps 32 -> 512+ across a 4096-token prefill, so the grid runs from 1x32=32
   blocks to 4x32=128. The 64-head loop then serialises inside those few
   blocks: the head axis is parallelism being spent as latency.

Entrpi's equivalent is `indexer_mxf4_encode_rows_kernel` (12.83 ms) +
`indexer_scores_mxf4_kernel` (4.94 ms) = 17.8 ms against our 92.38 ms.

## The two facts that make this cheap

**The K side is free.** From `pulsar_cuda_indexer.cu`: *"The cache rows are
QAT-roundtripped to exactly these values in both modes, so the scores are
bit-identical; packed mode only changes storage and read traffic."* The
compressed rows already sit exactly on the E2M1 x E8M0 grid. Feeding them to a
block-scaled MMA natively loses **nothing** — it is the same value the
dequantiser reconstructs.

**We already run the instruction.** `pulsar_mxfp4_cutlass.cu` uses the SM120
block-scaled `kind::f8f6f4` MMA in production for the MoE — cute atom
`SM120_16x8x32_TN<e2m1, e4m3, f32>`, i.e. **mixed E4M3 activations against E2M1
weights**, which is exactly the operand pair this scorer wants. And
`pack_act_e4m3_rowmajor` already packs activations to MXFP8/E4M3.

So this is not a new capability. It is pointing a proven path at a second
consumer.

## Design

### Operand assignment

    S[t,h,c] = q[t,h,:] . k[c,:]          (128-dim inner product)
    score[t,c] = ( sum_h ReLU(S[t,h,c]) * w[t,h] ) * scale

- **B = compressed rows**, native E2M1 + E8M0. Exact, zero conversion.
- **A = Q**, packed to E4M3 + E8M0 per 32-element block.
- Accumulate f32.

Both operands are K-major, which is what the `_TN` atom wants.

### Why E4M3 for Q and not E2M1

Pure mxf4 would be marginally faster per instruction, but **the kernel is not
math-bound** — the analysis above says staging, syncs and occupancy are the
cost. On the `f8f6f4` atom, k=32 per instruction either way, so E4M3 costs no
extra instructions. Spending Q's mantissa for throughput we would not collect
is strictly worse. E4M3 gives 3 mantissa bits plus a per-32 E8M0 scale against
E2M1's 1 bit.

This is the one lossy edge in the design: Q drops from `__half` (10 mantissa
bits, what the current kernel uses) to E4M3. See *Fidelity* below.

### Tiling

M is the **(token, head) product**, head-minor: row `t*64 + h`. This folds the
head loop into the GEMM instead of wrapping it.

- **M-tile = 128 rows = 2 tokens x 64 heads.** This is deliberate: the SM120
  block-scaled SF atom spans 128 rows (`Blk_MN`), and every tile must start on
  a 128-row boundary. Two tokens' worth of heads lands on that boundary
  exactly, with no padding and no gather.
- **N-tile = 128 compressed rows.** 128 rows x 68 B = 8.7 KB of B in shared,
  against the current 32 KB fp16 `b_sh` — 3.8x less, and no unpack ALU.
- **K = 128** (head_dim), four 32-element scale blocks per row, matching the
  MMA's SF granularity exactly.

Grid becomes `(ceil(n_comp/128), ceil(n_tokens/2))`. At n_comp=512,
n_tokens=512 that is 4 x 256 = **1024 blocks** against today's 128.

### Epilogue

The output tile is [128 M x 128 N]. Because M is head-minor, each token's 64
heads are 64 contiguous rows, so the reduction `sum_h ReLU(S) * w[t,h]` is a
64-row segmented reduction within the tile, producing 2 score rows per tile.
Apply `* scale`, then the causal mask (`comp >= (pos0+token+1)/ratio -> -INF`)
exactly as today.

One `__syncthreads()` per tile instead of 128 per block.

### CORRECTION: two claims above were wrong

Read from `cute/arch/mma_sm120.hpp` and `cute/atom/mma_traits_sm120.hpp`:

**1. `mxf8f6f4` does NOT consume our packed rows verbatim.** The atom
`SM120_16x8x32_TN_VS<e4m3, e2m1, f32, ue8m0, 32>` declares
`BRegisters = uint32_t[2]` — 64 bits for 8 E2M1 elements per thread, i.e.
**8 bits per element, byte containers**. The nibble->byte spread does not go
away; it only gets cheaper (a shift/mask, no table lookup, no scale multiply,
no float conversion) and the tile halves to 16 KB instead of 32 KB fp16.

**2. There IS a truly packed path**, and it is a genuine design fork:

| | `mxf8f6f4` (E4M3 x E2M1) | `mxf4nvf4` 2X (E2M1 x E2M1) |
|---|---|---|
| instruction | `m16n8k32` | `m16n8k64` |
| A registers | `uint32_t[4]` = 16 elem, byte containers | `uint32_t[4]` = 32 elem, **packed nibbles** |
| B registers | `uint32_t[2]` = 8 elem, byte containers | `uint32_t[2]` = 16 elem, **packed nibbles** |
| K tile smem (128x128) | 16 KB (spread) | **8 KB (verbatim)** |
| Q precision | **E4M3, 3 mantissa bits** | E2M1, 1 mantissa bit |

Both apply UE8M0 scales in hardware at VS=32, matching our 4 scales/row
exactly. Both are enabled for SM120/SM121 at CUDA >= 12.8
(`CUTE_ARCH_MXF8F6F4_MMA_ENABLED`, `CUTE_ARCH_MXF4NVF4_2X_UE8M0_MMA_ENABLED`),
so both are reachable at our `sm_120f`.

**Recommendation: build `mxf8f6f4` (E4M3 Q) first.** The kernel is not
math-bound, so `mxf4nvf4`'s k=64 and 8 KB tile buy less than they appear to,
while E2M1 Q throws away two mantissa bits on the only lossy operand. Keep
`mxf4nvf4` as a measured follow-up, and let the probe below decide it on
evidence rather than on this argument.

### SF thread mapping — the highest-risk part

Not a memory swizzle after all; the scales ride in registers, one byte per
thread per MMA. The mapping is non-obvious — from `mma_traits_sm120.hpp`:

    SFALayout = Layout<Shape<Shape<_2,_2,_8>,_32>>   // effectively 16 threads (2:0 mode)
    SFBLayout = Layout<Shape<Shape<_4,_8>,_32>>      // effectively  8 threads (4:0 mode)

So SFA is supplied by effectively 16 distinct lanes and SFB by 8, with threads
sharing scales through stride-0 modes. Guessing this is exactly the failure the
MoE comment warns about ("reads scrambled scales").

### Implementation route

Use the **cute atoms and their `MMA_Traits`** (`ALayout`/`BLayout`/`SFALayout`/
`SFBLayout`, plus the `fp4_shift_A`/`fp4_shift_B` helpers), not hand-written
PTX. The instruction string is easy to copy; the per-thread fragment and SF
layouts are not, and they are where this goes wrong silently. That means the
new kernel lives in its own TU that includes CUTLASS (as
`pulsar_mxfp4_cutlass.cu` does) rather than inside `pulsar_cuda_indexer.cu`,
which keeps CUTLASS out of the indexer's compile time.

### MEASURED instruction contract (tests/idx_mxfp4_probe.cu, all phases pass)

Every line below is measured on GB10, not read off a spec. Two of them
contradict the obvious reading and would each have produced silently wrong
scores.

| item | measured |
|---|---|
| availability | ships at our normal `sm_120f` — `pulsar_mxfp4_cutlass.o` holds 256 x `QMMA.SF.16832.F32.E4M3.E2M1.E8` |
| A (e4m3) | one value per byte at bit 0; standard table, exact |
| B (e2m1) | nibble at **bits [5:2]** of its byte, one per byte; decodes to **4x** the e2m1 table |
| ue8m0 | value = **2^(byte - 128)**, NOT the OCP spec's 127 bias |
| our cache -> hw | `sf_hw = stored_byte - 1` (+1 for the bias, -2 for the 4x) |
| SFA lanes | row m <- lane `4*(m%8) + (m/8)` (16 distinct lanes) |
| SFB lanes | col n <- lane `4*n` (8 distinct lanes) |
| C layout | row = `lane/4` for d0,d1 and `+8` for d2,d3; col = `(lane%4)*2 + {0,1}` |

The SF lane maps match cute's "effectively 16 threads" / "effectively 8
threads" comments exactly, which is a useful independent check on both.

**Two traps worth naming.** The ue8m0 bias is 128 here, so using the spec's 127
scales every product by 1/4 per operand — a silent 16x on a two-sided product,
which presents as "the k-sum is a quarter of what it should be" rather than as
an error. And the e2m1 nibble at bit 0 decodes as a LINEAR `code*0.5` ramp
(0,.5,1,1.5,2,2.5,3,3.5) that agrees with the real table for codes 0-4 and
diverges only at 5,6,7 — so a test using only small codes would pass and ship
wrong values for the top of the range.

**Build note.** Raw block-scaled asm cannot be compiled unguarded at
`-arch=sm_120f`: nvcc also emits it into the `compute_120` PTX fallback pass,
where ptxas rejects `.kind::mxf8f6f4`. CUTLASS guards its copy behind
`CUTE_ARCH_MXF8F6F4_MMA_ENABLED`, which is why it ships fine. The kernel must
do the same, or be built for a single arch-specific target.

### Step 0: a numerics probe before any kernel

Standalone, modelless, following the precedent of the MoE's self-check against
a double-precision oracle:

1. DONE — availability confirmed, and the contract above recovered.
2. DONE — SF thread mapping pinned empirically (table above).
3. DONE — see FORK DECISION below.

Gate at **shape time only**, never per token or per layer (see
`[[no-hot-path-flags]]`): `head_dim == 128 && n_head == 64 && !quality_mode`,
with an env flag to restore the fp16 WMMA path for A/B.

## Fidelity

Not bit-exact, and the change is **not** where it first appears to be:

- K: exact (QAT grid), no change.
- Q: `__half` -> E4M3. This is the whole numerical delta.
- Head-reduction order changes (segmented tree vs the current serial `h=0..63`
  accumulate). The file already accepts ~1 ULP for exactly this class of
  change in its warp-reduction comment.

The scorer's **only** consumer is top-k selection, so the metric that matters
is not score error but **selected-set overlap**. With top_k=512 out of
n_comp ramping to ~1024 we select roughly half, so boundary churn will be
non-zero by construction — but rows at the boundary are by definition the
least-weighted ones. That argument is plausible, not proven, and must be
measured rather than assumed.

Required before this goes near the engine:

1. **Top-k set overlap** vs the fp16 path across the n_comp ramp (32 -> 512+),
   not at a single point. The ramp is where the 2026-08-07 attention harness
   found four wrong shapes.
2. **End-to-end logits KL** against the A1 reference — the suite-v1 run that is
   already outstanding for the MMQ fold-order work.
3. `mixed_neutrality_gate` and `algo_stability_gate` for regressions.
4. A **calibrated** speed harness following `tests/attn_indexed_bench.cu`:
   `#include` the shipped `.cu` so it drives the real file-static kernel, with
   a gate that brackets the engine's observed per-launch range. A synthetic
   bench measuring a different regime than the engine is the documented cause
   of the 2026-07-22 MoE NO-GO.

## Expected return

92.38 ms today. Entrpi's equivalent stage is 17.8 ms including its encode. If
this lands in the 15-25 ms band the saving is ~70 ms on a ~4600 ms prefill:

- **~4-5x on the stage**
- **~1.5% end to end**

Worth stating plainly: the end-to-end number is small. The case for doing it is
that the surface is tiny and self-contained next to an attention rewrite, the
instruction and the packer already exist and are proven on this GPU, and the
K-side loss is provably zero. The Q-side packing cost must be budgeted against
the win — Entrpi pays 12.83 ms for its row encode, and a Q packer will not be
free either.


## FORK DECISION: E4M3 for Q (measured, tests/idx_quant_fidelity.cc)

Host-only simulation, no GPU: the fork is a quantisation question, not an MMA
question. K is held exactly on the E2M1 x E8M0 grid (which is what the cache
already stores), Q is quantised per-32 with an E8M0 scale, and the whole
`sum_h ReLU(q.k) * w` reduction is evaluated in f64 as reference.

**fp16 is effectively lossless for selection** (~100% of the exact top-512
retained), so it is the honest baseline: the churn below IS the change versus
what ships today.

| Q distribution | fp16 | E4M3 | E2M1 | E4M3 dropped rank (mean / worst) | E2M1 dropped rank |
|---|---|---|---|---|---|
| gaussian s=1 | 100.00% | **98.68%** | 95.43% | 501.0 / 470 | 479.6 / 382 |
| gaussian s=0.25 | 99.98% | **98.80%** | 95.41% | 501.4 / 476 | 474.9 / 377 |
| heavy-tail t3 | 100.00% | **98.71%** | 94.51% | 501.7 / 471 | 471.0 / 281 |
| heavy-tail t1.5 | 100.00% | **98.93%** | 95.31% | 502.5 / 473 | 473.7 / 333 |
| uniform [-1,1] | 100.00% | **98.36%** | 95.31% | 498.3 / 446 | 468.8 / 346 |

**E4M3 wins on both axes, and the margin is stable across every distribution**,
so the conclusion does not rest on having guessed Q's real shape.

The second number is the one that matters. The design argued that churn would
be confined to boundary rows -- the least-weighted of the selected set -- and
flagged that as plausible rather than proven. It is now measured:

- **E4M3 never drops a row ranked better than 446 of 512** across any trial or
  distribution, and its mean dropped rank is ~500, i.e. sitting on the
  selection boundary. The argument holds.
- **E2M1 drops rows from as high as rank 281** — mid-pack of the selected set,
  not the boundary. The argument FAILS for E2M1. Its 4.7% churn is also
  qualitatively worse than E4M3's 1.3%, not merely larger.

So `mxf4nvf4` (packed nibbles, k=64, 8 KB tile) is rejected on fidelity despite
its better data path, and the recommendation made on bottleneck grounds is
confirmed on measurement: **build `mxf8f6f4` with E4M3 Q**.

### What this does NOT establish

Synthetic Q. The sweep exists to show the answer is distribution-robust, and it
is, but it is not a substitute for the real thing. ~1.3% of the selected set
changing is small and lands on the boundary, yet the end-to-end consequence is
still owed: the suite-v1 KL run against the A1 reference, which is already
outstanding for the MMQ fold-order work. Nothing here licenses skipping it.

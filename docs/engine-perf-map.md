# Where prefill time actually goes

Measured 2026-08-08 on sparky (GB10, sm_121, 48 SMs), `pulsar-bench` at a
4001-token pure prefill (`--gen-tokens 0`) against
`/srv/models/v5mx4-0731-mmqaligned.gguf`, nsys `cuda_gpu_kern_sum`.

Reproduce:

    TMPDIR=<writable> nsys profile -o pk --trace=cuda --sample=none --cpuctxsw=none \
      ./pulsar-bench -m <model> --prompt-file <4k-token file> --ctx-max 4000 --gen-tokens 0
    nsys stats --report cuda_gpu_kern_sum --format csv pk.nsys-rep

`pulsar-bench` is the right harness, not the server: it is a short-lived pure
prefill, so the trace has no decode or HTTP noise, and it prints `prefill_tps`
directly.  It also carries `--dump-frontier-logits-dir`, which is how a change
claiming bit-exactness gets proven (`cmp` the JSON, not a tolerance).

## The map

| share | instances | avg      | kernel                                    |
|-------|-----------|----------|-------------------------------------------|
| 17.8% | 122       |  6.80 ms | `gateup_iq2_d2r_pair_kernel`               |
| 12.0% | 84        |  6.67 ms | `attention_indexed_mixed_heads8_online`    |
| 10.1% | 43        | 10.92 ms | `attention_static_mixed_heads8_online`     |
|  9.1% | 1290      |  0.33 ms | `cutlass3x_sm120_bstensorop_s16832gemm`    |
|  4.9% | 76        |  3.02 ms | cutlass grouped/array block-scaled GEMM    |
|  3.9% | 86        |  2.13 ms | `head_rms_norm_rope_tail_kernel`           |
|  3.8% | 172       |  1.03 ms | `rms_norm_plain_kernel`                    |
|  3.8% | 383       |  0.46 ms | `f32_to_f16_kernel`                        |
|  3.5% | 172       |  0.96 ms | `hc_expand_kernel`                         |
|  3.4% | 22        |  7.29 ms | `attention_decode_mixed_heads8_online`     |
|  3.1% | 58        |  2.51 ms | `pack_act_e4m3_rowmajor_warp`              |
|  2.6% | 86        |  1.38 ms | `mxfp8_quant_act_grouped_kernel`           |

## Ranked targets

### 1. Attention: 22.1% (indexed 12.0 + static 10.1), and it uses NO tensor cores

ncu on both kernels, three launches each, application replay (kernel replay
fails -- they run inside a CUDA graph, so `--replay-mode application` is
required):

    pipe_tensor   0%      <- zero
    pipe_fma      40%
    pipe_lsu      56%     <- top pipe
    pipe_alu       8%
    IPC           0.64 of 1.0

22% of prefill does its matmuls on the FP32 FMA pipe with the tensor cores
completely idle.  This is the largest single opportunity in the engine and it
is what the `flashinfer-attn` branch was named for.

It is NOT reachable by tuning.  Both occupancy knobs are already measured with
documented cliffs (`PULSAR_ATTN_MIN_BLOCKS` 2, where 3 spills and is 8x worse;
`PULSAR_ATTN_STATIC_MIN_BLOCKS` 4), the loads are already float4, and the
softmax is already online.  The limit is structural: head_dim is 512 with one
warp per head, so each of the 8 warps re-reads the whole 2 KB KV row out of
shared memory (8x amplification) and then spends a 32-lane reduction plus a
broadcast to produce ONE score from 16 FMAs of useful work.  A tensor-core
formulation removes both -- the reduction becomes the MMA's k dimension, and
operands are reused across the MMA's m dimension instead of being re-read per
head.  That is a FlashAttention-style rewrite, not a tweak.

#### The operand-format decision, measured

`tests/attn_precision_fidelity.cc` scores each candidate on REAL activations
(dumped with `PULSAR_DUMP_ATTN`, 24 tokens x 64 heads against an f64 reference),
and `tests/idx_mma_issue_bench.cu` measures what each one buys.

What it buys, measured on GB10:

| format | rate | vs the FP32 FMA pipe attention uses today |
|--------|------|-------------------------------------------|
| f32 FMA pipe | 14.5 TMAC/s | 1.0x (today) |
| f16 m16n8k16 | 62.9 TMAC/s | **4.3x** |
| bf16 m16n8k16 | 62.9 TMAC/s | **4.3x** |
| fp8 block-scaled | 125.0 TMAC/s | 8.6x |
| fp4 block-scaled | 251.4 TMAC/s | 17.3x |

What it costs, two independent activation sets (mean rel L2 / top-1 attention
position preserved / worst single head):

| format | set 1 | set 2 | set 2 max |
|--------|-------|-------|-----------|
| f32 (today) | 2.3e-7 / 100% | 1.7e-7 / 100% | 7.4e-7 |
| fp16 | 2.0e-4 / 100.00% | 5.8e-4 / 99.93% | 3.7e-3 |
| bf16 | 1.6e-3 / 99.87% | 4.9e-3 / 99.87% | 2.2e-2 |
| e4m3 unscaled | 3.0e-2 / 97.85% | 9.3e-2 / 96.81% | 4.4e-1 |
| MXFP8 | 3.0e-2 / 97.85% | 9.2e-2 / 96.81% | **1.00** |
| MXFP4 | 1.7e-1 / 87.30% | 3.8e-1 / 84.44% | **3.52** |

**bf16 is strictly dominated: identical throughput to fp16 (62.9 TMAC/s both)
for 3-8x the error.** There is no case for it. That half of the question has an
unambiguous answer and does not need judgement.

The remaining choice is fp16 (4.3x, top-1 preserved ~100%, KL <= 3e-7) against
fp8 (8.6x, but top-1 changes on 2-3% of head-token pairs and the worst single
head is off by 100%).  **Recommendation: fp16.**  Attention feeds everything
downstream, and doubling 4.3x to 8.6x is not worth a format whose worst case is
a head attending somewhere else entirely.

Two further findings:

- **Nothing saturates.**  max|q| = 17.58, max|kv| = 5.65, and zero elements clip
  in ANY candidate including e4m3 (limit 448).  This is purely a resolution
  question, not a dynamic-range one.
- **MX block scaling buys nothing here.**  MXFP8 is no better than unscaled
  e4m3 on the mean and WORSE on the worst case (1.00 vs 0.44).  Block scaling
  earns its keep when data spans many binades; this data does not, so the
  shared per-32 exponent only costs the smaller elements in each block some
  bits.  The device's block-scaled path has no fidelity advantage for attention
  -- which is not what the indexer's experience would have predicted.

Scope, honestly: both samples come from the raw-window path
(`pulsar_gpu_attention_prefill_raw_heads_tensor`), where MLA passes ONE latent
vector as both K and V -- so a format has to serve the score dot product and the
value sum at once, with no option to keep V wider.  The compressed path
(`n_comp > 0`) uses a separate `comp_kv` for the value and was NOT sampled.  The
two sets already differ by ~3x, so treat the absolute numbers as indicative and
the ORDERING as the result.

Expected gain, not overstated: attention is LSU-bound at 56% and FMA-bound at
only 40%, so 4.3x more math throughput does not become 4.3x end to end.  Much of
the win has to come from the tensor-core dataflow removing the 8x shared-memory
re-read (8 warps each pulling the whole 2 KB row), not from raw math.  2-3x on
the attention kernels is the realistic target, i.e. 11-17% of prefill.

#### Built: src/cuda/pulsar_cuda_attn_f16.cu (opt-in, PULSAR_CUDA_ATTN_F16)

fp16 chosen, kernel written, wired into both window launchers behind the env
flag.  MEASURED in-engine, same workload:

    attention_static_mixed_heads8_online   10.92 ms/launch   (43 launches)
    attn_f16_kernel, 1 M-tile               7.39 ms/launch   1.48x
    attn_f16_kernel, 2 M-tiles              6.39 ms/launch   1.71x
    cold prefill  868.4 -> 953.4 tok/s      +9.8%

Beware a comparison that looks better and is wrong: a standalone bench of the
new kernel at n_comp=0 runs 4.38 ms, which against 10.92 reads as 2.5x.  It is
not the same work -- the engine's calls carry ~512 compressed rows on top of
the 128-row raw window, so the standalone shape is a fifth of the rows.  1.48x
is the honest number.

The INDEXED path is now converted too, which is where most of the attention
time was (12.5% against the static path's 10.1%).  Same kernel, second mode:
compressed rows become a top-k SELECTION instead of a prefix and raw rows come
from a ring buffer, so the row plan is reproduced verbatim from the f32 kernel.
It is deliberately narrow -- banked descriptors and ATTN_PACK comp rows keep
the f32 kernel, because approximating a plan that picks the WRONG KV rows
produces plausible attention rather than an error.

    attention_static  10.1% (43 x 10.92 ms)  + indexed 12.5% (84 x 6.67 ms)
      = 22.6% of GPU, 1029 ms
    attn_f16_kernel   14.5% (127 x 4.81 ms)  =  611 ms      1.68x
    frontier 1 (cold)       869.1 -> 946.6 tok/s   +8.9%
    frontier 2 (continued)  770.5 -> 851.6 tok/s  +10.5%

Still on the FMA pipe: attention_decode_mixed_heads8_online (3.4%), plus the
banked/packed indexed variants this tier refuses.

Why not 4.3x, and what the second step bought.  ncu on the 1-M-tile build:
pipe_tensor 6-8%, pipe_lsu 33%, 7.15 GB of L2 traffic in 11.9 ms -- the MMAs
idled while the kernel moved KV.  Each block staged the whole window and four
head-groups per token staged the SAME rows (modelled 10.7 GB against 7.15
measured; the window ramps, so the model is high).

With M fixed at 16 the block count is n_tokens*n_head/16 NO MATTER how the 16
is split between tokens and heads, so batching tokens per block does not help
and only widens the staged window.  The only lever is more (token, head) PAIRS
per block -- more M-tiles -- which costs registers in both the Q fragments and
the O accumulator.  Two tiles halves the traffic and fits (114 regs, no spill);
four would need 128 before temporaries.

Two M-tiles bought 7.39 -> 6.39 ms, i.e. 1.16x from a 2x traffic cut, so
traffic was not the only limiter: at 512 threads only one block is resident, so
occupancy is still ~33%.  The remaining gap is worth another look, but it is no
longer obviously a bandwidth story.

End-to-end this IS a fidelity change, as designed: same greedy argmax, 9/10
top-10 overlap, mean |logit delta| 0.32 against a [-47.7, 37.5] range on the
frontier checked.  Default-on waits for the suite-v1 KL run.

#### Fidelity ledger: suite-v1 KL run (2026-08-08) — CLEARED, defaults flipped

Method note first, because the first attempt measured nothing: the KL harness
walks the doc token-by-token (decode path), so with the stock 32-token prefix
a candidate that only changes PREFILL produced bit-identical dumps — a
vacuous pass.  `--kl-prefix` (added for this run) makes the KV cache be built
by production-shaped prefill chunks; the walked logits then measure exactly
what a served request sees.  The dsml-tools doc (141 tokens, prompt below the
128-token chunk gate) stayed at KL ~1e-26 as the designed control.

Exact full-vocab KL(defaults || fp16+mxfp4), v5mx4-0731-mmqaligned, stride 4:

    depth ~1k (12 shorts, prefix 1024): mean 0.008-0.055, median ~2.4e-2,
                                        per-position p95 0.03-0.21
    depth  8k (long-mixed, prefix 8192):  mean 0.029, median 1.7e-2, p95 0.10
    depth 32k (long-mixed, prefix 32768): mean 0.013, median 3.2e-3, p95 0.06

Divergence SHRINKS with depth — the fp16 softmax error does not compound.

Against the full-fat vLLM reference (the one clean doc, see defect below):
defaults KL med 0.0011 / p95 0.390 / top-1 97.7%; candidate 0.0013 / 0.382 /
96.9%.  The quant's own divergence from source dominates; the flags moved
nothing outside noise.  Verdict: flipped both to default-on
(`pulsar_env_tier_on`, opt out with =0), closed-loop verified: no-env binary
reproduces the candidate dumps to 1e-26, =0 binary reproduces the baseline.

KNOWN DEFECT, blocks the wider cross-rig ledger: the ref-DeepSeek-V4-Flash-0731
capture let vLLM re-tokenize the suite TEXT, and on 12/14 docs it collapsed
special-looking spans into single ids — the reference model was conditioned on
a different token stream (pulsar matches the frozen suite tokens exactly on
all 14).  Re-capture on the work rig force-feeding the frozen token ids
(capture script change) before trusting any per-doc vs-reference number
beyond short-00/dsml-tools.

### The 2k decode deficit (2026-08-09): diagnosed, and the cheap fixes are dead

The one cell Entrpi still wins (19.85 vs 18.41 tok/s at 2k) is NOT overhead:
decode is 97.8% GPU-busy, one ~500us sampling gap per step (<1%).  The step
is 50.1 ms of kernel time and the slack is concentrated in
`attention_decode_mixed_heads8_online_kernel`: 7.5 ms/step (15%) from
grid (1, 8) — EIGHT blocks on a 48-SM chip, each walking all n_score rows.
Everything else in the step is at or near the 273 GB/s roofline.

Two hypotheses built, measured, and KILLED (kernels were bit-exact, reverted):

- Finer head split (more blocks, fewer heads each): void by inspection —
  every block still walks the full row list; the serial chain length is
  unchanged.  Never benched because the code read disproved it.
- Prefetch pipelining (group g+1's loads into registers during group g's
  scoring; tried 4-row and 8-row groups): 18.41 -> 18.37 -> 18.31, i.e.
  noise.  ncu: 1.71 active warps/scheduler, 57% long-scoreboard.  The
  decisive observation: per-ROW cost is ~287 ns across BOTH group sizes
  (160x1.17us == 80x2.3us), so the cost scales with rows, not with groups —
  the chain is per-row (score -> 5-deep shuffle reduction -> broadcast ->
  online update, plus a load round trip per row with ~2 warps/scheduler and
  40 idle SMs offering zero interleave).  No staging trick attacks that.

Conclusion: the ONLY fix is fewer rows per block — split-KV (flash-decoding)
with a softmax merge: e.g. 8 row-splits x 8 head-groups = 64 blocks, chain
shortens ~8x, est. 7.5 -> ~1.5-2 ms/step => ~+11% decode at 2k (flips the
cell) and gains at every depth.  This is the same merge machinery the
flash-style prefill rewrite wants; build it once, spend it twice.

#### SHIPPED (2026-08-09): split-KV decode, default-on

One kernel, two modes: gridDim.z > 1 makes each z-block walk its slice and
emit (m, l, o) partials to static device scratch (graph-safe, 16.8 MiB);
`attention_decode_split_merge_kernel` folds them log-sum-exp style and
applies the sink once.  gridDim.z == 1 is bit-identical to the old walk and
still serves n_tokens > 8.  All three heads8 sites route through one
dispatcher; PULSAR_CUDA_DECODE_SPLITKV=0 opts out (bit-identical restore,
verified 1e-26).  PULSAR_SPLITKV_DEBUG=1 A/Bs both walks per call.

Measured, locked clocks, v5mx4-mmqaligned (baseline -> split):
    2k  18.41 -> 20.72  (+12.5%, flips the last Entrpi-winning cell: 19.85)
    4k  17.48 -> 17.80   8k 17.22 -> 17.52   16k 17.02 -> 17.48
Past 2k the indexed carve-out serves the ratio-4 layers, so heads8's share
(and the win) shrinks — expected.

Fidelity: reassociation class, NOT bit-exact.  Unit gate
(tests/attn_decode_split_test.cu): split+merge vs single-walk across all
staging branches and descriptor modes, worst 1.1e-6 rel L2.  Engine KL
(cache-compounded teacher forcing): 2.9e-3..4.6e-2 by doc — same class as
the two PREVIOUSLY ACCEPTED reassociations (fp16 tier 0.008..0.055; the
heads8 carve-out itself scores 1.86e-2 on the same lens).  Vs full-fat
reference (clean doc): KL med 0.0016 / p95 0.388 / top-1 96.9% — noise
against defaults (0.0011 / 0.390 / 97.7%).  Spec verify batches ride the
split path; spec-on greedy text is identical split-on vs split-off (the
"need/should" near-tie flip between spec modes is pre-existing batched-GEMM
numerics, confirmed by control).

    Compute (SM) 58.5%, Memory 48.6%, L2 48.6%
    pipe_lsu 44%, pipe_alu 31%, pipe_tensor 19%, pipe_fma 15%

Balanced and already respectable, so headroom is ~1.7x at absolute best.  Note
`pipe_alu` (31%) runs well ahead of `pipe_tensor` (19%): the kernel spends more
issue on unpacking IQ2 than on the tensor math it feeds.  That is inherent to
MMQ, so this is the hardest of the three to move.

### 3. The 9 grouped-CUTLASS MoE layers cost more per layer than the 34 MMQ ones

`MoE expert tier: 9/43 layers grouped-CUTLASS, 34/43 per-expert-tiled`.
Attributing the kernels each tier launches gives roughly 1.0% of GPU time per
CUTLASS layer against 0.57% per MMQ layer -- call it ~4% of prefill if those 9
layers were type 43 instead of type 40.

Treat that number as soft.  The 1290-instance `cutlass3x` GEMM serves dense/MLA
projections across all 43 layers, not just the 9 MoE ones, so it cannot be
billed cleanly to the tier; the estimate above excludes it.  Confirming this
properly needs a model requantized to emit type 43 for those layers, which is a
PrismaQuant change, not a kernel change.

## Concurrency (measured 2026-08-08, servers, ctx 8192)

Same client load against both servers, one at a time: N concurrent chats with
DISTINCT prompts (so warm-reuse cannot serve one request from another), 200
greedy tokens each, aggregate = total completion tokens / wall.  Ours ran 5
auto-sized banks; the donor ran with --batched-session 12, its best case.

  N     ours plain   ours +DSpark   donor plain (12 sessions)
   1       19.1          24.8            14.9
   4       29.7          30.4            19.4
   8        --           30.1            19.4
  12       29.9          30.1            19.4

Ours leads aggregate serving throughput by ~54% at every batch level.  BOTH
engines saturate at N=4 and are flat to N=12 -- extra resident sessions buy
nothing on this load.  Speculation moves only the single-stream number
(spec_max_live=1), not the batch.

Two honesty notes.  First, the initial run of ours was accidentally spec-ON --
the v5mx4 gguf EMBEDS a DSpark drafter, so "no drafter found (gguf/dspark.gguf)"
does not mean plain; the later "DSpark drafter found in model" line does.
--no-dspark is required for a plain run, and the plain rerun changed N=1 from
24.8 to 19.1 while leaving the batch numbers alone.  Second, the donor README
claims 59 tok/s at 12 concurrent; on this load profile it measured 19.4.  The
claim is not reproduced here -- different workload, likely spec-on with a
drafter and possibly shared prefixes -- so treat the 54% as the like-for-like
result and the README figure as unverified on this box.

### Speculation vs batching: neither fork speculates in a batch

The donor hard-disables speculation whenever batched mode is active
(ds4_server.c: the decode loop gates on !batched_mode, and boot logs "MTP
speculative decoding is disabled while native session batching is active") --
even with one live stream.  Ours keeps speculation for ONE live session
(spec_max_live=1), which is why single-stream-through-the-server was 24.8 with
the drafter while the donor's equivalent cannot exceed its plain rate.  Neither
fork speculates across concurrent sessions; if DSpark's ~+30% single-stream
gain applied to 4-5 batched sessions it would stack on the 54% aggregate lead.
The hard part is that a rejected draft desyncs its session from the coalesced
batch step.  For OUR fork that limit is a confirmed deliberate decision
(2026-08-08), not an oversight -- spec_max_live=1 is where it was intentionally
stopped.  Treat lifting it as a design proposal to bring to Tyler, not a bug
fix.

## Measured dead ends -- tried on 2026-08-08, do not retry without new information

- **The startup "gap" against the donor engine was a misread, and our startup
  is FASTER.**  His boot log's "8.20 GiB in 1.70s" line is only the dense-span
  prep; a tail-windowed read missed the 20.9s his engine spends REBUILDING
  78.71 GiB of aligned SoA expert artifacts on 6 CPU threads at every process
  start.  Totals on the same box, same day: ours ~17s of model prep (21.4s
  wall including the bench), his ~24s (29.6s wall).  The mmqaligned GGUF is
  the reason -- it bakes the SoA layout at quantize time, so load is a plain
  16s copy instead of a repack.  Bake-at-quantize was the right design.
- **Every alternative to the startup staging copy measured WORSE on GB10:**
    staging copy (shipped)     ~17s prep   956/854 tok/s
    cudaMemPrefetch migration   66s prep   916/841   (driver faults every page
                                                      on the calling thread)
    madvise + ATS direct map    ~0s prep   102/623   (GPU first-touch ATS
                                                      faults land in prefill,
                                                      and recur every process)
  cudaHostRegister on the 86 GiB mapping returns "operation not supported" for
  BOTH MAP_SHARED and MAP_PRIVATE file-backed mappings on this driver, so the
  registration route is closed regardless of mmap flags.  The GPU-visible copy
  has to be made once somewhere; the staging copy makes it at 4.4 GiB/s up
  front, which is the cheapest of the three places to pay it.

- **Double-buffering the fp16 attention KV tile buys nothing.**  Implemented,
  measured, REVERTED.  ncu said the kernel is latency-bound (IPC 0.15-0.19 of
  1.0, no pipe above 27%, 33% occupancy), so overlapping the staging looked
  like the obvious fix.  It is not: the staging code BLOCKS ON ITS OWN LOADS.
  It is load -> wait -> store, straight-line, so moving it ahead of the compute
  does not overlap anything -- it still fully serialises before phase 1.
  942.7/849.7 against 942.0/845.3 tok/s, i.e. nothing, for 18 KB more smem and
  a second buffer's worth of complexity.
  Real overlap needs one of two things this kernel cannot currently have:
  cp.async, which copies BYTES and so cannot do the f32->fp16 conversion the
  staging performs; or holding the loaded values in registers across the
  compute, which is ~16 more registers against 119 of a 128 budget at 512
  threads.  Making the KV cache fp16 in GLOBAL memory would unlock cp.async and
  is the real fix, but that is an engine-wide format change, not a kernel one.

- **The elementwise / normalisation tier is already at the roofline.**  L2 bytes
  over kernel time: `f32_to_f16` 241 GB/s, `rms_norm_plain` 245,
  `head_rms_norm_rope_tail` 246 -- all ~90% of the ~273 GB/s the device has.
  Together they are ~11% of prefill and NONE of it is recoverable by making the
  kernels faster; only removing whole passes would do it, and the f32->f16 and
  MXFP8 conversions are already cached and armed per activation
  (`mxfp8_act_cache_t`, "1 quantization + 1 conversion instead of 2 + 5").
  Note an analytical estimate put `head_rms_norm_rope_tail` at 128 GB/s and was
  wrong by 2x -- it moves 544 MB, not the 268 MB the obvious arithmetic gives.
  Measure the bytes, do not derive them.
- **There is no scheduling or launch-overhead win.**  Sum of all kernel time is
  4.651 s against 4.970 s of prefill wall: the GPU is 93.6% busy.  Everything
  left is inside a kernel.
- **`--prefill-chunk` is already optimal at its default.**  8000-token prompt:
  1024 -> 673 tok/s, 2048 -> 728, 4096 (default) -> 729, 8192 -> 728.
- **cuBLASLt algo autotune buys nothing.**  Implemented, measured, reverted.
  Asking `cublasLtMatmulAlgoGetHeuristic` for 12 candidates and timing each on
  the real operands returns only **2-3** candidates, because the
  `REDUCTION_SCHEME_NONE` determinism constraint already prunes the set that
  hard.  The tuner picks a different algo than the heuristic's top choice about
  half the time, and throughput is IDENTICAL on cached shapes (792 vs 793,
  772 vs 771 tok/s); on the dominant shape it re-picks the heuristic's own #0.
  Net end-to-end it is slower, because each new `ntok` pays the tuning cost.
  The heuristic is right; do not go looking here again.
- **Why the biggest cuBLASLt GEMM sits at 31.8% of tensor peak.**  Its shape is
  in=1024, out=32768, ntok=2048: 137 GFLOP in 1.93 ms is 71 TFLOP/s of ~250,
  but it writes a 32768x2048 **f32** result -- 268 MB, i.e. ~139 GB/s of pure
  output store, half the device's bandwidth spent on the write.  The limiter is
  the output dtype, not the algo or the tile.  Narrowing it is a fidelity
  decision, so it belongs with the attention precision question, not before it.

## What is already at the roofline -- do not re-open

- `pack_act_e4m3_rowmajor_warp`: 260 GB/s of ~273 GB/s after the warp-per-
  4-blocks rewrite (was 154).  ~95% of bandwidth; nothing left.
- `mxfp8_quant_act_kernel` and its grouped twin: already warp-per-block with a
  shuffle reduction.  They look like the pack did but are not the same bug.
- The indexer scorer GEMM: 35 TMAC/s against a measured 125 TMAC/s instruction
  ceiling, and the whole indexer complex is ~2% of prefill.  See
  `docs/indexer-mxfp4-scorer.md`.

## Method note

Two measurement traps cost real time on 2026-08-08, both worth repeating:

1. **Sweep the independent variable before optimizing.** The indexer scorer was
   tuned for several rounds against a total runtime that was 82% Q pack.  A
   two-minute n_comp sweep separated fixed cost from per-row cost and showed the
   GEMM was already competitive; every "TMAC/s" figure before that sweep was
   total MACs over a runtime that was mostly something else.
2. **Lock the clock before comparing configs.** Unlocked, one identical binary
   measured 1.376e-4 and 1.178e-4 ms/row on two runs -- a 17% spread across
   differences being resolved at the 5% level, and it picked the wrong winner
   twice.  `sudo nvidia-smi -lgc 2600`, then `-rgc` afterwards.

## TEB comparison + the "quality loss" investigation (2026-08-09)

tool-eval-bench v2.5.1.dev11, 84-scenario hardmode, temp 0, seeds 42/43/44,
both engines on their own artifacts, same protocol.

Evals: Entrpi 85/85/85 (identical to the category, history-free).  Ours
86/79/75 on one long-lived server -- NOT nondeterminism and NOT decay.  A
fresh server reproduces seed 42's 86 exactly; 8 hardmode-only passes across
seeds x cache states are all identical (P=17/30); an aged server moved
hardmode P UP (21/30).  Outcomes are deterministic given (request, server
HISTORY): warm bank forking (route -> FORK-partial, 87% of eval requests)
resumes thinking checkpoints whose visible context differs from a cold
build, so borderline agentic trajectories land differently either way.
Coupling vector confirmed by falsification: disk KV cache OFF changes
nothing; kernels exonerated (split-KV scratch included).  This is a SERVING
DESIGN property (thinking-checkpoint reuse), not an engine numerics bug --
whether evals should pin behavior (exact-match forks only / fork-off flag)
is a product decision.

The remembered "92 on v0.1.0": not reachable by any current config we
measured (best cell 86); v0.1.0 = different artifact (pre-type-43), older
TEB scoring -- treat as incommensurable until someone re-runs that exact
combination.

Perf (TEB llama-benchy, pp2048/tg128, aggregate tg t/s, his -> ours):
    d0:    c1 20.2 -> 19.7 | c2 17.1 -> 23.1 | c4 16.0 -> 26.2
    d4096: c1 16.8 -> 24.7 | c2 11.7 -> 20.7 | c4 10.2 -> 23.9
    d8192: c1 16.6 -> 24.5 | c2  9.8 -> 17.3 | c4  8.1 -> 17.7
Opposite concurrency shapes: his falls with c (spec gated off batched),
ours rises.  His one cell: c1/d0 by 0.5 t/s.  Same-day locked-clock bench
sweep (his latest b030961, unchanged since Aug 5): we lead decode at every
depth (+5.4%..+8.7%) and prefill 2k-16k; his 32k prefill edge 1.1% is
inside his run-to-run variance.

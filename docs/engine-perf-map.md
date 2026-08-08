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

### 2. MoE gate/up: 17.8% in one kernel

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

## Measured dead ends -- tried on 2026-08-08, do not retry without new information

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

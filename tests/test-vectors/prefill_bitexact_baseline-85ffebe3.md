# prefill_bitexact_baseline-85ffebe3.bin — provenance

The committed baseline for `make cuda-prefill-gate`. This blob RE-ANCHORS the
gate: it supersedes `prefill_bitexact_baseline-055239b.bin`, whose anchor the
L037 attention levers deliberately broke.

- **Anchor ref:** `85ffebe3` — the merge that landed the L037 levers (cp.async
  double-buffered KV staging + the fused `q_prep` in the attention Q-fragment
  build) onto dev. That merge is the numerics change, so it is the anchor.
- **Why re-anchored (the decision, so nobody re-litigates it):** the levers
  are **correct per the f64 oracle but NOT bit-exact** against the old anchor
  — they reorder accumulation inside the fp16 attention tier, which is the
  deliberately-approximate tier. Measured **+3.0% prefill** (928.5 → 956.5
  t/s, 3 interleaved runs per arm vs the levers' own base `9cea2b7`, spreads
  <±0.3%, no overlap). Tyler accepted the trade 2026-08-20 (ledger L037).
- **Suite evidence at the anchor point:** 14/15 gates PASS on `85ffebe3`
  against the OLD baseline with `cuda-prefill-gate` the single FAIL — i.e.
  the only thing that moved was the thing the levers move. After re-anchoring:
  prefill gate PASS, provenance check OK.
- **Dumped:** 2026-08-20, on sparky (GB10, sm_121/sm_120f, CUDA 13.3, CUTLASS
  4.7.0), by `make cuda-prefill-gate-baseline PREFILL_BASELINE_REF=85ffebe3`
  in `~/l076-tree`.
- **Artifact:** `/srv/models/v5mx4-0731-srcfmt-v1-reapfix-lt.gguf` (the gate's
  `ds4flash.gguf` symlink). The blob's own header pins the model — a different
  artifact fails the header compare, not the byte compare.
- **sha256:** `8105f051e475ce1bd8cd29bb5b4fdf49446e4836b75a2ecb4e7af1b1e86cd8bb`

The previous blob stays in the tree: it is the record of the pre-lever
numerics, and `PREFILL_BASELINE_REF=055239b` still reproduces that comparison
on a build from before the merge.

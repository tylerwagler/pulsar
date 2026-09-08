# prefill_decode_baseline-4d0172e.bin — provenance

The committed baseline for `make cuda-prefill-decode-gate` (L181): the full-vocab
logits of the FIFTH classic decode step after prefilling the story blob to each of
the depths 1001, 4102 and 8197.  Magic `DS4PFXD1`; otherwise the prefill blob's format.

- **Anchor ref:** `4d0172e` — L210: routed IQ2 gate/up at decode takes a k-major GEMV
  tier (exact f32 weights, split-K fma order) instead of the MMA tile, on DECODE rows
  only (L167's row-kind predicate, whole M-neutral range, never on prefill rows).
  Deliberate numerics change confined to decode steps: the prefill blob is untouched
  (the 4d0172e prefill dump is byte-identical to the 684fa8d blob's payload), so
  `PREFILL_BASELINE_REF` stays at 684fa8d and only this anchor moves.  Graded where the
  reference gate could see the arithmetic (the tier's earlier cut on the 6-row prefill
  remainder at story depth 4102): 2.425e-07 vs 1.975e-07, under the gate's 1e-6 noise
  floor (L211), NET UNCHANGED.
- **Dumped:** 2026-09-08, on sparky (GB10, sm_120f, CUDA 13.3), by
  `make cuda-prefill-decode-gate-baseline PREFILL_DECODE_BASELINE_REF=4d0172e` in `~/pulsar-bank`.
- **Size / sha256:** 1551448 bytes / `c811d0460316d70cc8ffc078daf97786444ac563940d6a5d8ff236f65f6b11ba`
- **Superseded blob removed in the same commit:** 684fa8d (decode blob only).

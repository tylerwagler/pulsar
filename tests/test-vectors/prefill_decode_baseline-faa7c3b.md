# prefill_decode_baseline-faa7c3b.bin — provenance

The committed baseline for `make cuda-prefill-decode-gate` (L181): the full-vocab
logits of the FIFTH classic decode step after prefilling the story blob to each of
the depths 1001, 4102 and 8197.  Magic `DS4PFXD1`; otherwise the prefill blob's format.

- **Anchor ref:** `faa7c3b` — L210 target 2: decode attention is split-K flash-decode on
  the fp16 tile (4 logical splits per row, physical blocks sized to the device, combine
  folds the sink and the heads' inverse tail rope), on DECODE rows only (L167's row-kind
  predicate).  Deliberate numerics change confined to decode steps -- a reassociated
  softmax; the prefill blob is untouched (the prefill gate passes against the 4d0172e-era
  684fa8d blob unchanged), so `PREFILL_BASELINE_REF` stays at 684fa8d and only this
  anchor moves.  Graded against the previous decode anchor (4d0172e): KL 3.1e-8 / 3.0e-8 /
  5.3e-9 at the three depths, top-1 identical, top-5 overlap 4/5/4 -- the class of the
  landed expert tier's own move (1.1e-8 / 5.9e-8 / 8.1e-9).  rows/L210.md.
- **Dumped:** 2026-09-08, on sparky (GB10, sm_120f, CUDA 13.3), by
  `make cuda-prefill-decode-gate-baseline PREFILL_DECODE_BASELINE_REF=faa7c3b` in `~/pulsar-bank`.
- **Size / sha256:** 1551448 bytes / `e207a5ee7298f7fa3aa5331965291850a34014d01a70cc5b02dbb42ddbfba3f9`
- **Superseded blob removed in the same commit:** 4d0172e (decode blob only).

# prefill_decode_baseline-7b6e79a.bin — provenance

The committed baseline for `make cuda-prefill-decode-gate` (L181): the full-vocab
logits of the FIFTH classic decode step taken after prefilling the story blob to
each of the depths 1001, 4102 and 8197 -- every one of them leaves an unaligned
partial group in the ratio-4 compressor state, which the prefill's own frontier
logits never read.  Five steps, not one: the partial group is read only when it
closes (up to three steps later) and its pooled row is attended by the step
after that; one decode step was byte-blind to a misplaced partial group.
Magic `DS4PFXD1`; otherwise the prefill blob's format.

- **Anchor ref:** `7b6e79a` — dev after L177 ("the speculative lane's row budget is
  the M-neutral range, 16").  Anchored separately from the prefill blob: the
  decode steps' numerics move when the prefill's do not, and vice versa.
- **Dumped:** 2026-09-05, on sparky (GB10, sm_120f, CUDA 13.3), by
  `make cuda-prefill-decode-gate-baseline PREFILL_DECODE_BASELINE_REF=7b6e79a`
  in `~/pulsar-bank` with the five-step gate source.
- **Mutation-validated:** the rebuild placing the partial rows one slot off
  their phase fails the gate at depth 1001 (the whole-prompt rebuild path;
  worst logit off by 4.5) and cannot reach 4102 / 8197, whose unaligned final
  chunk is built by the per-row loop; a decode update that skips the ratio-4
  window shift is the mutation for those depths (see rows/L181.md).  The
  unmutated tree is byte-identical at all three depths.
- **Size / sha256:** 1551448 bytes / `e34b4f80a24454652357440d167cd4d6c561d86004dc31aa36447014d43a36a1`

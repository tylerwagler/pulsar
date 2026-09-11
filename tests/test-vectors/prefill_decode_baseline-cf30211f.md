# prefill_decode_baseline-cf30211f.bin — provenance

The committed baseline for `make cuda-prefill-decode-gate` (L181): the full-vocab
logits of the FIFTH classic decode step after prefilling the story blob to each of
the depths 1001, 4102 and 8197.  Magic `DS4PFXD1`; otherwise the prefill blob's format.

- **Anchor ref:** `cf30211f` — L219 B1: the routed IQ2 DOWN tensor takes the
  k-major GEMV at decode (the L210 tier that gate/up already had), replacing the
  N-starved D2R tile on decode rows.  Exact f32 weights and split-K fma order
  instead of the tile's E4M3-rounded weights.  A deliberate numerics change
  confined to decode steps: prefill rows still take the tile, and
  `cuda-prefill-gate` passes byte-identical against the 684fa8d blob unchanged.
- **Graded against the previous decode anchor (faa7c3b)** with a blob-to-blob
  KL/argmax comparison (all 129,280 vocabulary logits per depth):
    depth 1001: top-1 identical (304),   top-5 5/5, KL 1.26e-9 / 1.46e-9
    depth 4102: top-1 identical (28428), top-5 5/5, KL 9.24e-8 / 1.23e-7
    depth 8197: top-1 identical (34011), top-5 4/5, KL 6.67e-9 / 4.37e-9
  (KL is `old||new` / `new||old` after softmax.)  The worst raw `|dlogit|` is
  3.5-4.6, but the exact-byte gate headlines the max absolute logit difference;
  distributionally the move is 2-4 orders of magnitude smaller -- the same
  reassociation class as the previously landed decode anchors (faa7c3b itself
  graded 3.1e-8..5.3e-9 against 4d0172e), and the top-1 is stable at every depth.
- **Measured:** locked-clock decode +1.8% @2k / +2.0% @8k against d6d3feb
  (`pulsar-bench`, teacher_forced_corpus, 128 greedy tokens, two rounds).
- **Dumped:** 2026-09-11, on sparky (GB10, sm_120f, CUDA 13.3), by
  `make cuda-prefill-decode-gate-baseline PREFILL_DECODE_BASELINE_REF=cf30211f`
  in `~/pulsar-bank`.
- **Artifact:** the gate's `ds4flash.gguf` symlink (the blob's own header pins
  the model — a different artifact fails the header compare).
- **Size / sha256:** 1551448 bytes / `f4c4686d42c92022497f4d2c9e1b09ac9db70168f92c32656f9385cccba82df5`
- **Superseded blob removed in the same commit:** faa7c3b.

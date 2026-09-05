# prefill_bitexact_baseline-bae614d.bin — provenance

The committed baseline for `make cuda-prefill-gate` (see the gate section of
the top-level Makefile and `tests/prefill_bitexact_gate.cpp`).

- **Anchor ref:** `bae614d` — dev after L172 ("the CUB top-k kernel ranks by
  the same order relation as the bitonic kernels") and L173 ("one indexer
  scorer for every row count").  The last deliberate numerics changes at the
  time of the dump; both change the selected compressed rows past 4096
  compressed rows, so no older anchor can agree at depth 16388.
- **Depths:** 512, 2048, 4096, 4102, 6144, 8196, 16388 (L175: 8196 and 16388
  are one row past the n_comp 2048 and 4096 ranking-kernel buckets, each with
  a 4-row final chunk).  GATE_CTX 16512.
- **Dumped:** 2026-09-04, on sparky (GB10, sm_120f, CUDA 13.3), by
  `make cuda-prefill-gate-baseline PREFILL_BASELINE_REF=bae614d` in
  `~/pulsar-bank` with this branch's gate source.
- **Artifact:** the gate's `ds4flash.gguf` symlink (the blob's own header
  pins the model — a different artifact fails the header compare).
- **Size / sha256:** 3619928 bytes / `b49ff99e7b807ac8da4f1edc008ce94315045d74378e24553bf03c3c2a16dd7a`
- **Superseded blobs removed in the same commit:** every earlier anchor's
  blob (055239b .. f2b7912); they are in history, and the gate can only ever
  check against the anchor the Makefile names.

Self-describing format (`DS4PFXG1` magic): ref stamp, logits width, depth
list, FNV-1a of the prompt token ids, then the raw f32 logits per depth.

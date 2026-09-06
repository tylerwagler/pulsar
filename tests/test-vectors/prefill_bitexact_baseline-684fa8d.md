# prefill_bitexact_baseline-684fa8d.bin — provenance

The committed baseline for `make cuda-prefill-gate` (see the gate section of
the top-level Makefile and `tests/prefill_bitexact_gate.cpp`).

- **Anchor ref:** `684fa8d` — L195: the plain-weight prefill GEMM arm and the
  attn-out 'a' arm each run ONE cuBLASLt kernel per shape (chosen at 4096 rows),
  and the attention heads have ONE E4M3 encoding (the fused epilogue and the rope
  tail quantise the stored bf16 value the read-back encoder sees).  Deliberate
  numerics change: every prefill logit moves; graded CLOSER to the B300 reference
  (story net -25.2%, code net -58.2% over the confident depths, 2026-09-06).
- **Depths:** 512, 2048, 4096, 4102, 6144, 8196, 16388.  GATE_CTX 16512.
- **Dumped:** 2026-09-06, on sparky (GB10, sm_120f, CUDA 13.3), by
  `make cuda-prefill-gate-baseline PREFILL_BASELINE_REF=684fa8d` in `~/pulsar-bank`.
- **Artifact:** the gate's `ds4flash.gguf` symlink (the blob's own header pins
  the model — a different artifact fails the header compare).
- **Size / sha256:** 3619928 bytes / `c3c997416a0f36ec4931b924b539a7365d8f07d229d241b36d0c1bd1978deb87`
- **Superseded blob removed in the same commit:** bae614d.

Self-describing format (`DS4PFXG1` magic): ref stamp, logits width, depth
list, FNV-1a of the prompt token ids, then the raw f32 logits per depth.

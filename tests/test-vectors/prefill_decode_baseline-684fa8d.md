# prefill_decode_baseline-684fa8d.bin — provenance

The committed baseline for `make cuda-prefill-decode-gate` (L181): the full-vocab
logits of the FIFTH classic decode step after prefilling the story blob to each of
the depths 1001, 4102 and 8197.  Magic `DS4PFXD1`; otherwise the prefill blob's format.

- **Anchor ref:** `684fa8d` — L195 (see prefill_bitexact_baseline-684fa8d.md): the
  prefill's bytes moved, so the decode steps that read its KV moved with them.
- **Dumped:** 2026-09-06, on sparky (GB10, sm_120f, CUDA 13.3), by
  `make cuda-prefill-decode-gate-baseline PREFILL_DECODE_BASELINE_REF=684fa8d` in `~/pulsar-bank`.
- **Size / sha256:** 1551448 bytes / `53aa376e57d60a94a61eeff7180e1dbbcee995d68de767cfcabe56d37f8dd9ef`
- **Superseded blob removed in the same commit:** 7b6e79a.

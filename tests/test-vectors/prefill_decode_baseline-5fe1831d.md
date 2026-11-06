# prefill_decode_baseline-5fe1831d.bin — provenance

The committed baseline for `make cuda-prefill-decode-gate`: one classic decode step after each UNALIGNED prefill (1001 / 4102 /
8197), the step's full-vocab logits byte-compared against this blob. The prefill's own frontier logits never see the
compressor state, ring or seed the prefill leaves behind; this gate does, which is why it is anchored separately from
`prefill_bitexact_baseline`.

- **Anchor ref:** `5fe1831d`, the CUDA 13.5 + CUTLASS v4.8.1 toolchain: same anchor, same grade and same acceptance as the prefill
  twin (`prefill_bitexact_baseline-5fe1831d.md`). All three depths moved.
- **Artifact:** the Vision-Exp container (the battery's `FRONTIER_MODEL`).
- **Depths:** 1001, 4102, 8197.
- **Dumped:** 2026-11-06, on a GB10 (sm_120f, CUDA 13.5, CUTLASS v4.8.1), by
  `make cuda-prefill-decode-gate-baseline PREFILL_DECODE_BASELINE_REF=5fe1831d`.
- **Size / sha256:** 1551448 bytes / `dbfecb32147939b1aee1f8a4f911bba28a1fe93fd57ac4edf4b730dee127acc0`

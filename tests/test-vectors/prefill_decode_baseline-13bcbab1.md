# prefill_decode_baseline-13bcbab1.bin — provenance

The committed baseline for `make cuda-prefill-decode-gate` — one classic decode
step after each UNALIGNED prefill (1001 / 4102 / 8197), the step's full-vocab
logits byte-compared against this blob.  The prefill's own frontier logits never
see the compressor state, ring or seed the prefill leaves behind; this gate does,
which is why it is anchored separately from `prefill_bitexact_baseline`.

- **Anchor ref:** `13bcbab1` — the L292 NVFP4 exact-divide fix, same anchor and
  same reason as the prefill twin (see `prefill_bitexact_baseline-13bcbab1.md`):
  the 0731 NVFP4 KV row's divides are IEEE now, a deliberate numerics change
  accepted by Tyler's 2026-10-13 override.
- **Artifact:** the Vision-Exp container (the battery's `FRONTIER_MODEL`).
- **Depths:** 1001, 4102, 8197.
- **Dumped:** 2026-10-13, on a GB10 (sm_120f, CUDA 13.3), by
  `make cuda-prefill-decode-gate-baseline PREFILL_DECODE_BASELINE_REF=13bcbab1
  FRONTIER_MODEL=<the Vision-Exp container>`, 0 build warnings.
- **Size / sha256:** 1551448 bytes /
  `5c94215d3c1730f74d62bf9bc7feff1d7b138add14479451158d83749b5ad199`
- **Superseded blob:** `prefill_decode_baseline-f5bea7aa.bin` is deleted with
  this re-anchor; `prefill_decode_baseline-cf30211f.bin` (the deleted
  0731 artifact's) was deleted 2026-10-30; its note is in docs/history/.

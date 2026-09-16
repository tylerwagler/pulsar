# prefill_decode_baseline-f5bea7aa.bin — provenance

The committed baseline for `make cuda-prefill-decode-gate` — one classic decode
step after each UNALIGNED prefill (1001 / 4102 / 8197), the step's full-vocab
logits byte-compared against this blob.  The prefill's own frontier logits never
see the compressor state, ring or seed the prefill leaves behind; this gate does,
which is why it is anchored separately from `prefill_bitexact_baseline`.

- **Anchor ref:** `f5bea7aa` — the dev tip `~/devref-clean` is parked on (same
  reason as the prefill twin: the artifact is what moved, not the arithmetic).
- **Artifact:** `/srv/models/v5-vexp-full256-iq2-t46.gguf` — the Vision-Exp
  0731 artifact the battery's `FRONTIER_MODEL` names.  See the prefill blob's
  `.md` for why the previous anchor made this gate uninformative.
- **Depths:** 1001, 4102, 8197.
- **Dumped:** 2026-09-15, on sparky (GB10, sm_120f, CUDA 13.3), by
  `~/devref-clean/tests/prefill_bitexact_gate /srv/models/v5-vexp-full256-iq2-t46.gguf
  --dump-decode FILE` — the reference binary again.
- **Size / sha256:** 1551448 bytes /
  `d31a745e7f5ba4e7450740c5ba551f21d452cd8a6009a2d98c6b99fdc18133e1`
- **Cross-check (2026-09-15):** this branch at `e8915426` checks against this
  blob on the Vision-Exp artifact with **3/3 decode depths byte-identical,
  129280 full-vocab logits each, PREFILL GATE PASS**.
- **Superseded blob:** `prefill_decode_baseline-cf30211f.bin` stays in the tree
  as the `ds4flash.gguf` artifact's reference; it is no longer the default.

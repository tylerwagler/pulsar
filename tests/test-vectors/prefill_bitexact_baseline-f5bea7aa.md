# prefill_bitexact_baseline-f5bea7aa.bin — provenance

The committed baseline for `make cuda-prefill-gate` (see the gate section of
the top-level Makefile and `tests/prefill_bitexact_gate.cpp`).

- **Anchor ref:** `f5bea7aa` — the dev tip `~/devref-clean` is parked on, i.e.
  the reference implementation this branch is graded against.  Same numerics
  anchor as `684fa8d` (L195's one-cuBLASLt-kernel-per-shape prefill arm and the
  single E4M3 attention-head encoding); what changed is the ARTIFACT, not the
  arithmetic.
- **Artifact:** `/srv/models/v5-vexp-full256-iq2-t46.gguf` — the Vision-Exp
  0731 artifact the battery's `FRONTIER_MODEL` names.  This is the whole point
  of the re-anchor: the previous blob was dumped through the `ds4flash.gguf`
  symlink, which on sparky resolves to
  `/srv/models/v5mx4-0731-srcfmt-v1-reapfix-lt.gguf`.  The gate's header pins
  only the logits width and the prompt FNV, so a battery run with the Vision-Exp
  artifact compared TWO DIFFERENT ARTIFACTS' logits and reported 129280/129280
  differing at every depth — a FAIL with no information in it.  A blob is only
  meaningful for the artifact it was dumped on.
- **Depths:** 512, 2048, 4096, 4102, 6144, 8196, 16388.  GATE_CTX 16512.
- **Dumped:** 2026-09-15, on sparky (GB10, sm_120f, CUDA 13.3), by
  `~/devref-clean/tests/prefill_bitexact_gate /srv/models/v5-vexp-full256-iq2-t46.gguf
  --dump FILE` — i.e. by the REFERENCE BINARY, not by this branch.  The gate
  source is identical in both trees, so the harness is the same and only the
  engine differs, which is the property the anchor needs.
- **Size / sha256:** 3619928 bytes /
  `75ecc866385c0fa173d956bd7ba45b13ad42eae80be0dff2ae2f629ada690fb6`
- **Cross-check (2026-09-15):** this branch at `e8915426` checks against this
  blob on the Vision-Exp artifact with **7/7 depths byte-identical, 129280
  full-vocab logits each, PREFILL GATE PASS**.  So the re-anchor is not grading
  the branch against itself: the bytes came from dev.
- **Superseded blob:** `prefill_bitexact_baseline-684fa8d.bin` stays in the tree
  (it is the reference for the `ds4flash.gguf` artifact and its own checked-in
  `.md` records that); it is simply no longer the default.

Self-describing format (`DS4PFXG1` magic): ref stamp, logits width, depth
list, FNV-1a of the prompt token ids, then the raw f32 logits per depth.

# prefill_bitexact_baseline-13bcbab1.bin — provenance

The committed baseline for `make cuda-prefill-gate` (see the gate section of
the top-level Makefile and `tests/prefill_bitexact_gate.cpp`).

- **Anchor ref:** `13bcbab1` — the tip of the L292 NVFP4 exact-divide fix
  (work/nvfp4-land: `57267e95` IEEE `__fdiv_rn` divides for the 0731 NVFP4 KV
  row in `pulsar_cuda_attnpack.cu` + `13bcbab1` the KV payload v4 / segment v5
  bump) on dev `ab567485`.  That branch is squashed into ONE commit on dev; the
  squash's engine sources are byte-identical to `13bcbab1`'s, so the anchor's
  arithmetic is the landed arithmetic.  A deliberate numerics change moves the
  anchor (rule 7): the exact divide makes the device KV row equal the host recipe
  in `tests/attn_pack_fixture.h`, and rarely changes a KV code.
- **Grade:** the reference gate graded this change a noise-level AWAY from an
  FP8-KV B300 capture that cannot arbitrate NVFP4 rounding; Tyler accepted it
  (2026-10-13 override, see the Makefile's cuda-reference-gate-budget note and
  ledger L292).
- **Artifact:** the Vision-Exp container (DeepSeek-V4-Flash Vision-Exp) that the
  battery's `FRONTIER_MODEL` names.  A blob is only meaningful for the artifact
  it was dumped on: the header pins the logits width and the prompt FNV, not the
  weights.
- **Depths:** 512, 2048, 4096, 4102, 6144, 8196, 16388.  GATE_CTX 16512.
- **Dumped:** 2026-10-13, on a GB10 (sm_120f, CUDA 13.3), by
  `make cuda-prefill-gate-baseline PREFILL_BASELINE_REF=13bcbab1
  FRONTIER_MODEL=<the Vision-Exp container>` (detached worktree at the ref,
  this tree's gate source + Makefile copied in), 0 build warnings.
- **Size / sha256:** 3619928 bytes /
  `209cb5310d7fb32bc279e9b2dd8c2aeafda582999831bd70b36f90f5e666efe2`
- **Superseded blob:** `prefill_bitexact_baseline-f5bea7aa.bin` (same artifact,
  the fast-divide KV row) is deleted with this re-anchor, as numerics re-anchors
  do.  `prefill_bitexact_baseline-684fa8d.bin` stays as the deleted 0731
  artifact's history.

Self-describing format (`DS4PFXG1` magic): ref stamp, logits width, depth
list, FNV-1a of the prompt token ids, then the raw f32 logits per depth.

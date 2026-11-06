# prefill_bitexact_baseline-5fe1831d.bin — provenance

The committed baseline for `make cuda-prefill-gate` (see the gate section of the top-level Makefile and
`tests/prefill_bitexact_gate.cpp`).

- **Anchor ref:** `5fe1831d`, the CUDA 13.5 + CUTLASS v4.8.1 toolchain (L354). `5d13f9c5` on CUDA 13.3 + the previous pin
  matched the previous blob at all 7 depths; `5fe1831d` differs at 4102, 8196 and 16388 and matches at 512, 2048, 4096 and
  6144. Nothing in the engine's sources between them changes arithmetic: the toolchain moved it. Which of nvcc, cuBLASLt
  and CUTLASS did was not split.
- **Grade:** against the B300 capture, only depth 4102 moved in the story anchors: KL 2.162e-06 -> 2.316e-06, top-1 matching
  everywhere. NET over the confident depths +0.78%, further from the source. The code anchors did not move. Tyler accepted
  it (2026-11-06: "we're not gonna stay pinned on an old toolchain. just regrade it and reset the baseline"). See the
  Makefile's note at cuda-reference-gate-budget.
- **Artifact:** the Vision-Exp container (DeepSeek-V4-Flash Vision-Exp) that the battery's `FRONTIER_MODEL` names. A blob is
  only meaningful for the artifact it was dumped on.
- **Depths:** 512, 2048, 4096, 4102, 6144, 8196, 16388.
- **Dumped:** 2026-11-06, on a GB10 (sm_120f, CUDA 13.5, CUTLASS v4.8.1), by
  `make cuda-prefill-gate-baseline PREFILL_BASELINE_REF=5fe1831d` (a detached worktree at the ref, this tree's gate source
  and Makefile copied in).
- **Size / sha256:** 3619928 bytes / `d6aec7e5c30f1170f6a20e9deecabfa02552ba9a456de249350285df5585e341`

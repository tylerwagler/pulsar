# prefill_bitexact_baseline-qwen-d3e488fe.bin — provenance

The committed baseline for `make cuda-prefill-gate`: the frontier logits after prefill (see the gate section of the top-level Makefile and `tests/prefill_bitexact_gate.cpp`).

- **Anchor ref:** `d3e488fe`, the build production serves (v0.5.0-rc2; CUDA 13.5, CUTLASS v4.8.1).
- **Artifact:** `ElytronSystems/Qwen-3.8-Flash-Next-pulsar-k4` at revision `f48fa82904404703185600392104969510fa11ff`, from the HF cache
  (`BASELINE_MODEL`, default `QWEN_GATE_MODEL`). A blob is only meaningful for the artifact it was dumped on; a new
  revision of the container needs a new blob (L357).
- **Depths:** 7 depths: 512, 2048, 4096, 4102, 6144, 8196, 16388 (the gate's own list), x 248320 logits.
- **Dumped:** 2026-11-07, on a GB10 (sm_120f), by `make cuda-prefill-gate-baseline` (a detached worktree at the ref, this tree's gate source
  and Makefile copied in), L357 step 4.
- **Size / sha256:** 6953048 bytes / `ef5432a3c7ca8803694b45fca46439932ed4f3609420814240a55da6863b6ab1`
- **Replaces:** the DeepSeek Vision-Exp blob at 5fe1831d. DeepSeek left the runner (Tyler 2026-11-07: smoke only).

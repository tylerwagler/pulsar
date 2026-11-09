# prefill_decode_baseline-qwen-d3e488fe.bin — provenance

The committed baseline for `make cuda-prefill-decode-gate`: the logits after 5 decode steps (see the gate section of the top-level Makefile and `tests/prefill_bitexact_gate.cpp`).

- **Anchor ref:** `d3e488fe`, the build production serves (v0.5.0-rc2; CUDA 13.5, CUTLASS v4.8.1).
- **Artifact:** `ElytronSystems/Qwen-3.8-Flash-Next-pulsar-k4` at revision `f48fa82904404703185600392104969510fa11ff`, from the HF cache
  (`BASELINE_MODEL`, default `QWEN_GATE_MODEL`). A blob is only meaningful for the artifact it was dumped on; a new
  revision of the container needs a new blob (L357).
- **Depths:** 3 depths: 1001, 4102, 8197 (the gate's own list), x 248320 logits.
- **Dumped:** 2026-11-07, on a GB10 (sm_120f), by `make cuda-prefill-decode-gate-baseline` (a detached worktree at the ref, this tree's gate source
  and Makefile copied in), L357 step 4.
- **Size / sha256:** 2979928 bytes / `5542faeefb426da4204df35b6bd5a05831392f27af3fba337681d309a13d0b88`
- **Replaces:** the DeepSeek Vision-Exp blob at 5fe1831d. DeepSeek left the runner (Tyler 2026-11-07: smoke only).

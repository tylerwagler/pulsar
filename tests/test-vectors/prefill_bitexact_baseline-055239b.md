# prefill_bitexact_baseline-055239b.bin — provenance

The committed baseline for `make cuda-prefill-gate` (see the gate section of
the top-level Makefile and `tests/prefill_bitexact_gate.cpp`). Committing it
is the L046 fix: before this, every gate PASS depended on one un-tracked blob
surviving in somebody's working tree, and a fresh clone could not run the
gate at all (the bootstrap target could not link the old anchor's vendor
tree against the current Makefile).

- **Anchor ref:** `055239b` — "kv: the drafter seed quantized twice too, so
  its rows could differ from the target's". The last deliberate numerics
  change at the time of the dump, which is the only correct anchor for a
  bit-exactness gate (see the Makefile's gate comment for why "oldest commit
  that loads the artifact" is the wrong rule).
- **Dumped:** 2026-08-19, on sparky (GB10, sm_120f, CUDA 13.3, CUTLASS
  4.7.0), by `make cuda-prefill-gate-baseline` in `~/pulsar-srcfmt`.
- **Artifact:** `/srv/models/v5mx4-0731-srcfmt-v1-reapfix-lt.gguf` (the
  gate's `ds4flash.gguf` symlink). The blob's own header pins the model —
  a different artifact fails the header compare, not the byte compare.
- **sha256:** `ca9d032d08f0a259a8a658e67ae247123a65d6da84f5619618fe7a5f02d3203b`
- **Verified:** the 2026-08-19 full gate suite (15/15 PASS at tree 49e4843)
  checked the current engine against exactly this blob: 129,280 full-vocab
  logits byte-identical at all five depths (512, 2048, 4096, 4102, 6144).

Self-describing format (`DS4PFXG1` magic): ref stamp, logits width, depth
list, FNV-1a of the prompt token ids, then the raw f32 logits per depth.
The gate's `--check` verifies the ref stamp against `PREFILL_BASELINE_REF`
and the model header before comparing a single byte, so a stale or
mismatched blob fails loudly.

Re-anchor procedure (whenever a numerics change ships, and only then):
update `PREFILL_BASELINE_REF` in the Makefile, run
`make cuda-prefill-gate-baseline` on the GB10 (it dumps straight to the new
tracked filename), and commit the new blob + its .md + the Makefile line in
one commit. Leave the old blob pair in place until nothing references it.

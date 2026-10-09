# b300/refs -- the reference captures we owe, in the same rental

`run_refs.sh` runs on two GPUs (MP = 2) while `run.sh` quantizes on the rest, then hands the two GPUs to the
quantization (`refs/GPUS_FREE`).

## Stack and pins

**DeepSeek's own `inference/` code, from each snapshot, unmodified** (no route-trace patch), under `torchrun
--nproc-per-node 2`, converted by the snapshot's own `convert.py --model-parallel 2` -- the V4.1 kit's stack
(`notes/reference-capture/v41/`: "dtype decisions follow this code").  Pins: python 3.12, torch 2.10.0+cu130,
tilelang 0.1.8, apache-tvm-ffi 0.1.8.post2 (tilelang's declared range resolves to a release that breaks its
import), transformers 5.19.0, safetensors 0.8.0, numpy 2.5.3 (`bootstrap.sh REFS=1`; resolved for x86_64 on
2026-10-09).  MP = 2 because every MP=2 sharding assert divides for V4.1 (MP = 4 / 8 do not: Engram table 1 has
384,016,682 rows).  Models: `deepseek-ai/DeepSeek-V4.1-Flash@2cba9e42aa026125f3ed06c6d98c1db82f7ca027` (the
quantizer's source), `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp@6821d6ad3681a4b137b066b76094fa82ebd0a380` (the
served model; `VEXP=1`).

**Go / no-go first** (`probe_fp8.py`, the kit's on-start probe as a file): the reference's own fp8 linear at
M = 1 / 512 / 2048 must be bit-identical on rerun, NaN-free and within 1e-2 of fp32.  The only V4.1 capture so far
(4 x B200, MP = 4, revision dba1be0a, 2026-09) failed exactly here -- tilelang 0.1.8's fp8 GEMM returned NaN rows
and different bits for M > 1 on sm_100 -- and has 160/643 NaN tensors.  B300 is sm_103 and untested with this
stack; a NO-GO stops the V4.1 captures (the quantization is unaffected).  The options then are Tyler's: vLLM's V4.1
image (`notes/reference-capture/v41/onstart_vllm.sh`, `vllm/vllm-openai:deepseekv41-flash-0909-cu129`, needs a
docker-capable host and a driver that does not exist yet), or a later tilelang.

## What it produces (`refs/out/`, packed by pack.sh under `refs/`)

| dir | from | files | read by |
|---|---|---|---|
| `v41/` | the kit's `ref_capture_v41.py` (phases logits, anchors, dspark, greedy) | `story.ref.bin` (DS4PFXG1 v2; depths 512, 2048, 3840, 4096, 4102, 6144, 8192, 30464 -- the gate's six plus 3840 and 8192, 8 = MAX_DEPTHS), `code.ref.bin` (512, 2048, 3840), `*.ref.json`, `*_manifest.json`, `*.top32.safetensors`, `*.full_logits_d2048.safetensors`, `*.anchors_d512.safetensors` (+ the 4102 tail), `*.dspark_d2048.*`, `*.greedy_d2048.*`, `capture_manifest.json`, `{story,code}.tokens.bin` | `cuda-reference-gate` / `decode-reference-gate` / `verify-width-probe` with `PULSAR_REF_DIR=<this dir>` once pulsar serves V4.1 (L269 W5 (c): the gate is SKIP for V4.1 until this exists); KL budgets for V4.1 do not exist yet (a depth without a budget is not a failure) |
| `v41-short/` | the kit, phase logits | `story.ref.bin` / `code.ref.bin` at 64, 128, 256 (+ sidecars, tokens) | `decode-reference-gate` / `verify-width-probe` with `--ref-dir` (or `PULSAR_REF_DIR`) pointed here: the short rows cannot join the 8-depth blob, and gates_runner hard-codes `<slug>.ref.bin`, so they are their own directory |
| `v41-images/` | `capture_extra.py --mode images` | per case `img-at40`, `img-at256`, `img-at300`, `img-straddle256`, `img-deep2200`, `img-straddle4096`, `img-late6000`: `<case>.tokens.bin` (the expanded ids), `<case>.ref.bin` (DS4PFXG1 v2: last-row logits at E, E+1, E+8, E+32, E = the image block's end, each a fresh one-pass prefill), `<case>.json` (block start / slots / ViT grid / token types / image sha256 / top-64 per depth); `images/synth-640x480.png` | **no gate yet**: the reference readers prefill token ids only.  L283's owed grade (an image block past the 4096 chunk boundary and one straddling it, the vision_chunk_gate positions) needs an engine mode that merges the PNG at `image_start`; these files carry everything it needs, in the gate's blob format |
| `vexp-short/` (`VEXP=1`) | `capture_extra.py --mode depths` | `story.ref.bin` / `code.ref.bin` at 64, 128, 256 + tokens + sidecars | the dsgrade follow-up (L284): `verify-width-probe` / `decode-reference-gate --ref-dir` -- **caveat:** the existing vexp references (`~/ref-vexp`) came from vLLM 0.28.1rc1.dev137 on 2 x H200 (bf16 head, fp8 KV); these rows come from DeepSeek's own stack.  They grade the short depths against the reference model; they are not the same stack as the long rows |
| `vexp-images/` (`VEXP=1`) | `capture_extra.py --mode images` | as v41-images, in Vision-Exp's layout (image slots are vocab_size + type, N-layout with pads) | as v41-images: the served model's image reference for L283 |

Every case re-uses the kit's token bins (`story.tokens.bin` 30,474 ids, `code.tokens.bin` 3,840 ids; never
re-tokenized) -- the image cases place the image into the story stream and expand it with the model's own
`prepare_vl_inputs`.  `refs/out/SHA256SUMS` covers everything.  `test_cases.py` checks the image cases' layout on
CPU against a snapshot (both models pass, 2026-10-09: 206 slots per block on V4.1, 209 on Vision-Exp).

## Not done here

- vLLM captures (the existing vexp references' stack): the host would need docker and the vLLM image; not part
  of this kit.
- The kit's `render` and `trace` phases (they need the reap corpus cases, which are not shipped).

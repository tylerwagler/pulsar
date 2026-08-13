# REAP-25 expert-survivor transplant (ds4 v5mx)

This directory vendors the REAP-25 routed-expert **survivor map** the v5mx build
uses, plus the transplant tool that stamps it onto our own quantized
intermediate.

## What is here

| File | Purpose |
|------|---------|
| `reap25-lcb50-survivors.json` | The vendored artifact: per-layer list of surviving original expert ids (+ policy / keep_count / layout). This is the **only external input** the REAP stage needs. |
| `recover_survivors.py` | Regenerates `reap25-lcb50-survivors.json` from the upstream GGUFs (range-fetches only the small router tensors, never the full weights). |

## Provenance & attribution

The survivor set is the **REAP-25 (LiveCodeBench-50-sample-calibrated)** prune of
DeepSeek-V4-Flash published by **eouya2** as
[`eouya2/DeepSeek-V4-Flash-REAP25-LCB50-DS4`](https://huggingface.co/eouya2/DeepSeek-V4-Flash-REAP25-LCB50-DS4)
(file `DeepSeek-V4-Flash-REAP25-LCB50-DS4-compact-IQ2XXS.gguf`). The prune keeps
192 of 256 routed experts on layers 3–42 (layers 0–2 are hash-routed and
preserved), top-k 6, using [REAP — Router-weighted Expert Activation
Pruning](https://arxiv.org/abs/2510.13999).

**Why this is a redistributable derived artifact, not weights.** The eouya2 repo
carries the HF tag `license: other` with no restrictive body text; its bundled
`ds4_reap_runtime/LICENSE` is MIT (ds4.c + ggml authors), and the underlying
DeepSeek-V4-Flash weights are MIT (`general.license = "mit"` in the GGUF KV). What
we vendor here is **derived integer index metadata** — *which* experts survive per
layer — recovered from the router tensors, containing **no model weights**. We
redistribute it with attribution. If eouya2 later attaches restrictive terms, drop
`reap25-lcb50-survivors.json` and run `recover_survivors.py` to regenerate it
locally instead (see docs/ARTIFACT_BUILD.md §Upstream provenance, which
documents the cite-as-dependency fallback).

## How the survivor ids were recovered

eouya2's GGUF stores only `reap.layer.policy` and `reap.layer.keep_count` in its
KV — **not** the survivor expert ids. The identities are implicit in the compacted
F16 router (`ffn_gate_inp.weight`) rows: each pruned layer keeps 192 of the
original 256 router rows byte-for-byte. `recover_survivors.py` matches those rows
against the full 256-row router in the GGUF eouya2 pruned from
(`antirez/deepseek-v4-gguf` → `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`,
byte-identical F16 routers) to recover each survivor's original expert id. All 40
pruned layers matched 192/192 exactly.

## Usage

The map is consumed directly by the quantizer, which trims expert tensors to
the survivors as it generates them. There is no separate transplant step:

```sh
deepseek4-quantize ... --reap-survivors gguf-tools/reap/reap25-lcb50-survivors.json
```

The template must be built with the same map so the shapes agree
(`build_main_template.py --reap-survivors ...`); the quantizer cross-checks the
expert counts and refuses a mismatch.

```sh
# Regenerate the vendored map (needs network + curl; ~140 MB of router fetches):
python3 gguf-tools/reap/recover_survivors.py
```

## Path-B layout note

The router (`ffn_gate_inp`) and bias (`exp_probs_b`) stay padded to the full
`n_expert = 256` so the CUDA `router_select` kernels (which hardcode 256) are
untouched; only the expert weight tensors are physically dense-trimmed to the
survivors. Padded router rows are zeroed and their paired bias set to `-1e30`, so
they can never win top-k. Survivors are indexed densely `0..K-1` in sorted
original-id order.
